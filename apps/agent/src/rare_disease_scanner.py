"""
Rare Disease Scanner

Loads a FAISS index of clinical case reports (indexed by disease name + UMLS CUI),
a MedEmbed embedding model, a UMLS->MONDO mapping file, and a biomedical knowledge
graph (kg.feather) to perform comprehensive rare disease differential scanning.

Usage:
    from src.rare_disease_scanner import scanner
    results = scanner.scan(case_summary="...", patient_symptoms=["fever", "rash"])
"""

import json
import pickle
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[4]
FAISS_INDEX_PATH = REPO_ROOT / "shareable" / "faiss_index" / "index.faiss"
FAISS_META_PATH = REPO_ROOT / "shareable" / "faiss_index" / "index.pkl"
RAG_METADATA_PATH = REPO_ROOT / "shareable" / "metadata.pkl"
MAPPING_PATH = Path(__file__).parent / "umls_to_mondo_mapping.json"
KG_PATH = REPO_ROOT / "kg.feather"

TOP_K = 10
MAX_UNIQUE_DISEASES = 5


class RareDiseaseScanner:
    """Singleton scanner that loads all data at init time for fast lookups."""

    def __init__(self, llm=None):
        self.llm = llm
        self._loaded = False

    def _load(self):
        if self._loaded:
            return

        print("[RareDiseaseScanner] Loading MedEmbed model on CPU...")
        self.model = SentenceTransformer("abhinand/MedEmbed-large-v0.1", device="cpu")

        print("[RareDiseaseScanner] Loading FAISS index...")
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "rb") as f:
            self.faiss_meta = pickle.load(f)

        print("[RareDiseaseScanner] Loading RAG metadata...")
        with open(RAG_METADATA_PATH, "rb") as f:
            rag_data = pickle.load(f)
        self.rag_texts = rag_data["texts"]
        self.rag_metadata = rag_data["metadata"]

        self.disease_to_chunks: dict[str, list[int]] = defaultdict(list)
        for i, meta in enumerate(self.rag_metadata):
            disease = meta.get("disease", "").strip().lower()
            if disease:
                self.disease_to_chunks[disease].append(i)

        print("[RareDiseaseScanner] Loading UMLS->MONDO mapping...")
        with open(MAPPING_PATH) as f:
            self.umls_to_mondo = json.load(f)

        print("[RareDiseaseScanner] Loading kg.feather...")
        self.kg = pd.read_feather(str(KG_PATH))

        self._build_pheno_parent_child()
        self._build_disease_symptoms_index()
        self._build_disease_drugs_index()

        self._loaded = True
        print("[RareDiseaseScanner] All data loaded successfully.")

    def _build_pheno_parent_child(self):
        """Build child -> parent mapping for symptom deduplication."""
        pp = self.kg[self.kg["relation"] == "phenotype_phenotype"]
        self.pheno_parent_child: dict[str, str] = {}
        for _, row in pp.iterrows():
            child = str(row["y_name"]).strip().lower()
            parent = str(row["x_name"]).strip().lower()
            self.pheno_parent_child[child] = parent

    def _build_disease_symptoms_index(self):
        """Pre-build disease_id -> set(symptom_names) from disease_phenotype_positive."""
        pheno = self.kg[self.kg["relation"] == "disease_phenotype_positive"]
        self.disease_symptoms: dict[str, set[str]] = {}
        for mondo_id, group in pheno.groupby("x_id"):
            raw_symptoms = set(str(s).strip() for s in group["y_name"].tolist())
            self.disease_symptoms[mondo_id] = self._dedup_symptoms(raw_symptoms)

    def _dedup_symptoms(self, symptoms: set[str]) -> set[str]:
        """Remove parent symptoms when their children are present."""
        lower_map = {s.lower(): s for s in symptoms}
        keep = set()
        for s_lower, s_orig in lower_map.items():
            if s_lower in self.pheno_parent_child:
                parent_lower = self.pheno_parent_child[s_lower]
                if parent_lower in lower_map:
                    continue
            keep.add(s_orig)
        return keep

    def _build_disease_drugs_index(self):
        """Pre-build disease_name -> list[{drug, relation}] from drug->disease rows."""
        drug_relations = [
            "indication",
            "contraindication",
            "drug_effect",
            "off-label use",
        ]
        drug_rows = self.kg[self.kg["relation"].isin(drug_relations)]
        self._disease_name_to_drugs: dict[str, list[dict]] = defaultdict(list)
        for _, row in drug_rows.iterrows():
            disease_name = str(row["y_name"]).strip().lower()
            drug_name = str(row["x_name"]).strip()
            relation = str(row["relation"]).strip()
            self._disease_name_to_drugs[disease_name].append(
                {"drug": drug_name, "relation": relation}
            )

    def _get_mondo_id(self, umls_cui: str) -> str | None:
        """Map UMLS CUI to MONDO numeric ID."""
        return self.umls_to_mondo.get(umls_cui)

    def _get_symptoms_for_mondo(self, mondo_id: str) -> set[str]:
        """Get symptoms for a MONDO ID, handling MONDO_grouped composite IDs."""
        if mondo_id in self.disease_symptoms:
            return self.disease_symptoms[mondo_id]

        if "_" in mondo_id:
            parts = mondo_id.split("_")
            all_symptoms = set()
            for part in parts:
                if part in self.disease_symptoms:
                    all_symptoms.update(self.disease_symptoms[part])
            return all_symptoms

        return set()

    def _get_drugs_for_disease(self, disease_name: str) -> dict:
        """Get drug relations for a disease name."""
        drugs = self._disease_name_to_drugs.get(disease_name.lower(), [])
        indications = [d["drug"] for d in drugs if d["relation"] == "indication"]
        contraindications = [
            d["drug"] for d in drugs if d["relation"] == "contraindication"
        ]
        return {"indications": indications, "contraindications": contraindications}

    def _retrieve_diseases(self, case_summary: str) -> list[dict]:
        """Embed case summary, search FAISS, return deduplicated disease matches."""
        embedding = self.model.encode([case_summary], normalize_embeddings=True).astype(
            "float32"
        )

        distances, indices = self.faiss_index.search(embedding, TOP_K)

        disease_scores: dict[str, dict] = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.rag_metadata):
                continue
            meta = self.rag_metadata[idx]
            disease = meta.get("disease", "").strip()
            umls = meta.get("umls", "").strip()
            if not disease:
                continue

            key = disease.lower()
            similarity = float(1.0 - dist)
            if key not in disease_scores or similarity > disease_scores[key]["score"]:
                disease_scores[key] = {
                    "disease": disease,
                    "umls": umls,
                    "score": similarity,
                    "chunk_indices": [],
                    "evidence_snippets": [],
                }
            disease_scores[key]["chunk_indices"].append(int(idx))

        for info in disease_scores.values():
            snippets = []
            for ci in info["chunk_indices"][:2]:
                text = self.rag_texts[ci]
                snippets.append(text[:300] + "..." if len(text) > 300 else text)
            info["evidence_snippets"] = snippets

        sorted_diseases = sorted(
            disease_scores.values(), key=lambda x: x["score"], reverse=True
        )
        return sorted_diseases[:MAX_UNIQUE_DISEASES]

    def _compute_symptom_overlap(
        self, patient_symptoms: list[str], disease_symptoms: set[str]
    ) -> float:
        """Compute Jaccard-like overlap between patient and disease symptoms."""
        if not disease_symptoms:
            return 0.0

        patient_lower = {s.lower().strip() for s in patient_symptoms if s.strip()}
        disease_lower = {s.lower().strip() for s in disease_symptoms}

        if not patient_lower:
            return 0.0

        matches = patient_lower & disease_lower
        return len(matches) / len(disease_symptoms)

    def _llm_judge(
        self,
        case_summary: str,
        patient_symptoms: list[str],
        disease_name: str,
        disease_symptoms: set[str],
        evidence_snippets: list[str],
    ) -> dict:
        """Use isolated LLM to judge if rare disease is a plausible consideration."""
        if self.llm is None:
            return {
                "judgment": "uncertain",
                "reasoning": "LLM not available for judgment.",
            }

        symptom_list = "\n".join(f"- {s}" for s in sorted(disease_symptoms)[:30])
        evidence = "\n\n".join(
            f"Snippet {i + 1}: {s}" for i, s in enumerate(evidence_snippets)
        )
        patient_symptom_list = (
            "\n".join(f"- {s}" for s in patient_symptoms)
            if patient_symptoms
            else "None provided."
        )

        system_prompt = textwrap.dedent("""\
            You are a clinical reasoning assistant evaluating whether a rare disease
            is a plausible consideration for a doctor to flag. This is NOT a diagnosis.

            Evaluate based on:
            1. Symptom overlap between the patient and the disease
            2. Clinical plausibility given the presentation
            3. Whether the disease pattern matches the patient's profile

            Output ONLY a JSON object with:
            - "judgment": one of "plausible", "implausible", "uncertain"
            - "reasoning": brief 1-2 sentence explanation

            Default to "implausible" unless there is genuine clinical overlap.
            Most cases will NOT have a rare disease.
        """)

        user_prompt = textwrap.dedent(f"""\
            PATIENT CASE SUMMARY:
            {case_summary}

            PATIENT SYMPTOMS:
            {patient_symptom_list}

            RARE DISEASE: {disease_name}

            DISEASE SYMPTOMS (from knowledge graph):
            {symptom_list}

            RAG EVIDENCE:
            {evidence}

            Is this rare disease a plausible consideration for the doctor to flag?
        """)

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = response.content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            result = json.loads(content)
            return {
                "judgment": result.get("judgment", "uncertain"),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            return {
                "judgment": "uncertain",
                "reasoning": f"LLM judgment failed: {e}",
            }

    def _classify_missing_symptoms(
        self,
        missing: list[str],
        disease_name: str,
    ) -> dict[str, list[str]]:
        """Classify missing symptoms as askable vs diagnostic."""
        askable = []
        diagnostic = []

        askable_keywords = [
            "family",
            "history",
            "pain",
            "ache",
            "fatigue",
            "weakness",
            "nausea",
            "dizziness",
            "headache",
            "blurred",
            "vision",
            "hearing",
            "numbness",
            "tingling",
            "shortness",
            "breath",
            "sleep",
            "appetite",
            "weight",
            "mood",
            "anxiety",
            "onset",
            "sudden",
            "gradual",
            "intermittent",
            "constant",
            "worse",
            "better",
            "morning",
            "night",
            "exercise",
            "fever",
            "chills",
            "sweat",
            "cough",
            "rash",
            "itch",
            "swelling",
            "stiffness",
            "cramp",
            "spasm",
            "tremor",
            "seizure",
            "fainting",
            "confusion",
            "memory",
            "speech",
            "walking",
            "balance",
            "coordination",
            "urine",
            "bowel",
            "menstrual",
            "pregnancy",
            "birth",
            "developmental",
        ]

        diagnostic_keywords = [
            "biopsy",
            "histology",
            "genetic",
            "mutation",
            "gene",
            "chromosome",
            "karyotype",
            "sequencing",
            "assay",
            "biomarker",
            "level",
            "concentration",
            "enzyme",
            "activity",
            "density",
            "scan",
            "imaging",
            "mri",
            "ct",
            "x-ray",
            "ultrasound",
            "electrophysiology",
            "electrocardio",
            "echocardiogram",
            "angiography",
            "culture",
            "serology",
            "antibody",
            "antigen",
            "protein",
            "metabolite",
            "urinalysis",
            "hemoglobin",
            "platelet",
            "white blood",
            "red blood",
            "creatinine",
            "calcification",
            "mineral",
            "ossification",
            "electroencephalo",
            "nerve conduction",
            "spirometry",
        ]

        for symptom in missing:
            s_lower = symptom.lower()
            is_diagnostic = any(kw in s_lower for kw in diagnostic_keywords)
            is_askable = any(kw in s_lower for kw in askable_keywords)

            if is_diagnostic and not is_askable:
                diagnostic.append(symptom)
            elif is_askable:
                askable.append(symptom)
            else:
                askable.append(symptom)

        return {"askable": askable[:10], "diagnostic_tests": diagnostic[:10]}

    def _generate_questions(self, askable_symptoms: list[str]) -> list[str]:
        """Generate natural language questions for askable symptoms."""
        questions = []
        for symptom in askable_symptoms:
            s = symptom.strip()
            if "family" in s.lower() or "history" in s.lower():
                questions.append(f"Do you have a family history of {s.lower()}?")
            elif "pain" in s.lower():
                questions.append(f"Have you been experiencing {s.lower()}?")
            elif "onset" in s.lower() or "sudden" in s.lower():
                questions.append(f"Did your symptoms have a {s.lower()} onset?")
            else:
                questions.append(f"Have you noticed {s.lower()}?")
        return questions[:5]

    def scan(
        self,
        case_summary: str,
        patient_symptoms: list[str],
        previous_results: dict | None = None,
    ) -> dict:
        """
        Run a comprehensive rare disease scan.

        Args:
            case_summary: PHI-stripped clinical case summary
            patient_symptoms: List of patient-reported/observed symptoms
            previous_results: Previous scan results (for re-scans with new info)

        Returns:
            Structured dict with rare disease matches, coherence scores,
            missing symptoms, and recommendations.
        """
        self._load()

        matches = self._retrieve_diseases(case_summary)

        results = []
        for match in matches:
            disease_name = match["disease"]
            umls_cui = match["umls"]
            mondo_id = self._get_mondo_id(umls_cui)

            if mondo_id is None:
                continue

            disease_symptoms = self._get_symptoms_for_mondo(mondo_id)
            if not disease_symptoms:
                continue

            overlap_score = self._compute_symptom_overlap(
                patient_symptoms, disease_symptoms
            )

            if overlap_score == 0:
                continue

            judgment = self._llm_judge(
                case_summary,
                patient_symptoms,
                disease_name,
                disease_symptoms,
                match["evidence_snippets"],
            )

            if judgment["judgment"] == "implausible":
                continue

            patient_symptom_set = {s.lower().strip() for s in patient_symptoms}
            disease_symptom_lower = {s.lower().strip() for s in disease_symptoms}
            matching = [
                s for s in disease_symptoms if s.lower().strip() in patient_symptom_set
            ]
            missing = [
                s
                for s in disease_symptoms
                if s.lower().strip() not in patient_symptom_set
            ]

            missing_classified = self._classify_missing_symptoms(missing, disease_name)

            drugs = self._get_drugs_for_disease(disease_name)

            results.append(
                {
                    "disease_name": disease_name,
                    "umls_cui": umls_cui,
                    "mondo_id": mondo_id,
                    "confidence": round(match["score"], 3),
                    "llm_judgment": judgment["judgment"],
                    "llm_reasoning": judgment["reasoning"],
                    "symptom_overlap_score": round(overlap_score, 3),
                    "matching_symptoms": matching[:10],
                    "missing_symptoms": missing_classified,
                    "relevant_treatments": drugs.get("indications", [])[:5],
                    "contraindicated_treatments": drugs.get("contraindications", [])[
                        :5
                    ],
                    "rag_evidence_snippets": match["evidence_snippets"],
                }
            )

        results.sort(key=lambda x: x["confidence"], reverse=True)

        has_askable = any(
            m["missing_symptoms"].get("askable")
            for m in results
            if m["llm_judgment"] in ("plausible", "uncertain")
        )

        askable_questions = []
        if has_askable:
            all_askable = []
            for m in results:
                all_askable.extend(m["missing_symptoms"].get("askable", []))
            askable_questions = self._generate_questions(all_askable[:8])

        plausible_count = sum(1 for m in results if m["llm_judgment"] == "plausible")
        uncertain_count = sum(1 for m in results if m["llm_judgment"] == "uncertain")

        if plausible_count > 0:
            scan_summary = (
                f"Rare disease scan identified {plausible_count} potential match(es) "
                f"worth considering"
            )
            if uncertain_count > 0:
                scan_summary += f", plus {uncertain_count} uncertain match(es)"
            scan_summary += "."
        else:
            scan_summary = (
                "Rare disease scan found no strong matches. "
                "No rare disease flags at this time."
            )

        recommendations = []
        for m in results:
            tests = m["missing_symptoms"].get("diagnostic_tests", [])
            if tests:
                recommendations.append(
                    f"For {m['disease_name']}: consider {', '.join(tests[:3])}"
                )

        return {
            "rare_disease_matches": results,
            "scan_summary": scan_summary,
            "recommendation": " ".join(recommendations)
            if recommendations
            else "No specific recommendations at this time.",
            "has_askable_symptoms": has_askable,
            "askable_questions": askable_questions,
            "plausible_count": plausible_count,
            "uncertain_count": uncertain_count,
        }


scanner = RareDiseaseScanner()
