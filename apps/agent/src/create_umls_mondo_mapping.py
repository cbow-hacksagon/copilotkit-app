"""
One-time script to generate UMLS CUI -> MONDO numeric ID mapping.

Uses two-tier approach:
1. Primary: SSSOM TSV (mondo_exactmatch_umls.sssom.tsv) for direct UMLS->MONDO mappings
2. Fallback: kg.feather disease name matching for CUIs not in SSSOM

Output: src/umls_to_mondo_mapping.json
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SSSOM_PATH = REPO_ROOT / "mondo_exactmatch_umls.sssom.tsv"
KG_PATH = REPO_ROOT / "kg.feather"
METADATA_PATH = REPO_ROOT / "shareable" / "metadata.pkl"
OUTPUT_PATH = Path(__file__).parent / "umls_to_mondo_mapping.json"


def load_sssom_mappings() -> dict[str, str]:
    """Load UMLS CUI -> MONDO numeric ID from SSSOM TSV."""
    df = pd.read_csv(SSSOM_PATH, comment="#", sep="\t")
    mapping = {}
    for _, row in df.iterrows():
        umls_cui = row["object_id"].replace("UMLS:", "")
        mondo_id = row["subject_id"].replace("MONDO:", "").lstrip("0")
        mapping[umls_cui] = mondo_id
    return mapping


def load_kg_disease_names() -> dict[str, str]:
    """Load disease name -> MONDO numeric ID from kg.feather."""
    kg = pd.read_feather(KG_PATH)
    diseases = kg[kg["x_type"] == "disease"][
        ["x_name", "x_id", "x_source"]
    ].drop_duplicates("x_name")
    name_to_mondo = {}
    for _, row in diseases.iterrows():
        name_lower = row["x_name"].strip().lower()
        mondo_id = row["x_id"]
        name_to_mondo[name_lower] = mondo_id
    return name_to_mondo


def load_rag_diseases() -> dict[str, str]:
    """Load UMLS CUI -> disease name from RAG metadata."""
    with open(METADATA_PATH, "rb") as f:
        rag_data = pickle.load(f)
    cui_to_disease = {}
    for meta in rag_data["metadata"]:
        umls = meta.get("umls", "")
        disease = meta.get("disease", "")
        if umls and disease:
            cui_to_disease[umls] = disease
    return cui_to_disease


def main():
    print("=== UMLS -> MONDO Mapping Generator ===")
    print()

    print("Step 1: Loading SSSOM mappings...")
    sssom_mapping = load_sssom_mappings()
    print(f"  Loaded {len(sssom_mapping)} SSSOM mappings")

    print("Step 2: Loading RAG metadata...")
    cui_to_disease = load_rag_diseases()
    unique_cuis = set(cui_to_disease.keys())
    print(f"  Found {len(unique_cuis)} unique UMLS CUIs in RAG metadata")

    print("Step 3: Loading kg.feather disease names...")
    kg_disease_names = load_kg_disease_names()
    print(f"  Found {len(kg_disease_names)} unique disease names in kg.feather")

    merged = {}
    sssom_hits = 0
    fallback_hits = 0
    unmapped = []

    for cui in unique_cuis:
        if cui in sssom_mapping:
            merged[cui] = sssom_mapping[cui]
            sssom_hits += 1
        else:
            disease = cui_to_disease.get(cui, "").strip().lower()
            if disease and disease in kg_disease_names:
                merged[cui] = kg_disease_names[disease]
                fallback_hits += 1
            else:
                unmapped.append((cui, cui_to_disease.get(cui, "unknown")))

    print()
    print("=== Results ===")
    print(f"  SSSOM direct mappings:    {sssom_hits}")
    print(f"  kg.feather name fallback: {fallback_hits}")
    print(
        f"  Total mapped:             {len(merged)} / {len(unique_cuis)} ({100 * len(merged) / len(unique_cuis):.1f}%)"
    )
    print(f"  Unmapped:                 {len(unmapped)}")

    if unmapped:
        print()
        print("=== Unmapped CUIs (sample) ===")
        for cui, disease in unmapped[:10]:
            print(f"  {cui} -> {disease}")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print()
    print(f"Mapping written to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
