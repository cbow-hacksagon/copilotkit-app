"use client";
import { useState, useRef, useCallback } from "react";
import { useCoAgent } from "@copilotkit/react-core";

interface AgentImage {
  id: number;
  base64: string;
  mimeType: string;
  description: string;
}

interface AgentState {
  Imaging?: AgentImage[];
}

export function ImageChatPopup() {
  const [open, setOpen] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [base64, setBase64] = useState<string | null>(null);
  const [mimeType, setMimeType] = useState<string | null>(null);
  const [description, setDescription] = useState("");
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { state, setState } = useCoAgent<AgentState>({
    name: "default",
    initialState: { Imaging: [] },
  });

  const images = state.Imaging ?? [];

  const processFile = useCallback((file: File) => {
    setError(null);

    if (!file.type.startsWith("image/")) {
      setError(`Invalid file type: ${file.name}. Only images are accepted.`);
      return;
    }

    if (file.size > 5242880) {
      setError(`File too large: ${file.name}. Maximum size is 5MB.`);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setPreview(result);
      // extract mime type and raw base64 separately
      const [dataPrefix, rawBase64] = result.split(",");
      const mimeMatch = dataPrefix.match(/data:(image\/\w+);base64/);
      setMimeType(mimeMatch ? mimeMatch[1] : "image/png");
      setBase64(rawBase64);
    };
    reader.onerror = () => setError(`Failed to read ${file.name}`);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };

  const handleSubmit = () => {
    if (!base64 || !description.trim()) return;
    setSubmitting(true);

    const newImage: AgentImage = {
      id: images.length + 1,
      base64,
      mimeType: mimeType || "image/png",
      description: description.trim(),
    };

    setState({ Imaging: [...images, newImage] });

    // reset form
    setPreview(null);
    setBase64(null);
    setMimeType(null);
    setDescription("");
    setSubmitting(false);
    setOpen(false);
  };

  const clearImage = () => {
    setPreview(null);
    setBase64(null);
    setMimeType(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <>
      {/* Trigger button - bottom-right corner FAB */}
      <button
        onClick={() => setOpen((p) => !p)}
        aria-label="Upload image"
        style={{
          position: "fixed",
          bottom: "24px",
          right: "24px",
          width: "48px",
          height: "48px",
          borderRadius: "50%",
          background: "var(--primary)",
          color: "var(--primary-foreground)",
          border: "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000,
          boxShadow: "0 2px 12px rgba(0,0,0,0.18)",
          transition: "opacity 0.15s ease",
        }}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="2"/>
          <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
          <path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        {images.length > 0 && (
          <span style={{
            position: "absolute",
            top: "-4px",
            right: "-4px",
            background: "var(--destructive)",
            color: "var(--destructive-foreground)",
            fontSize: "10px",
            fontWeight: 700,
            width: "18px", height: "18px",
            borderRadius: "50%",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            {images.length}
          </span>
        )}
      </button>

      {/* Popup */}
      {open && (
        <div style={{
          position: "fixed",
          bottom: "84px",
          right: "24px",
          width: "min(380px, calc(100vw - 32px))",
          borderRadius: "14px",
          border: "0.5px solid var(--border)",
          background: "var(--card)",
          zIndex: 999,
          overflow: "hidden",
          boxShadow: "0 8px 32px rgba(0,0,0,0.14)",
        }}>
          {/* Header */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "12px 16px",
            borderBottom: "0.5px solid var(--border)",
          }}>
            <span style={{ fontSize: "13px", fontWeight: 500, color: "var(--foreground)" }}>
              Add image to agent
            </span>
            <button
              onClick={() => { setOpen(false); clearImage(); setDescription(""); }}
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted-foreground)", padding: "2px" }}
            >
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                <path d="M1 1l11 11M12 1L1 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </button>
          </div>

          <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: "12px" }}>

            {/* Error message */}
            {error && (
              <div style={{
                padding: "8px 10px",
                borderRadius: "8px",
                background: "var(--destructive)",
                color: "var(--destructive-foreground)",
                fontSize: "12px",
                fontWeight: 500,
              }}>
                {error}
              </div>
            )}

            {/* Drop zone / preview */}
            {preview ? (
              <div style={{ position: "relative" }}>
                <img
                  src={preview}
                  alt="preview"
                  style={{
                    width: "100%",
                    maxHeight: "180px",
                    objectFit: "cover",
                    borderRadius: "8px",
                    border: "0.5px solid var(--border)",
                    display: "block",
                  }}
                />
                <button
                  onClick={clearImage}
                  style={{
                    position: "absolute", top: "6px", right: "6px",
                    width: "22px", height: "22px",
                    borderRadius: "50%",
                    background: "rgba(0,0,0,0.55)",
                    border: "none", cursor: "pointer",
                    display: "flex", alignItems: "center", justifyContent: "center",
                  }}
                >
                  <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
                    <path d="M1 1l7 7M8 1L1 8" stroke="#fff" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>
            ) : (
              <div
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                style={{
                  padding: "28px 16px",
                  border: `1.5px dashed ${dragging ? "var(--ring)" : "var(--border)"}`,
                  borderRadius: "8px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: dragging ? "var(--accent)" : "transparent",
                  transition: "all 0.15s ease",
                }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                  style={{ margin: "0 auto 8px", display: "block", color: "var(--muted-foreground)" }}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  <polyline points="17 8 12 3 7 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <line x1="12" y1="3" x2="12" y2="15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
                <p style={{ fontSize: "12px", color: "var(--muted-foreground)", margin: 0 }}>
                  Drop image or{" "}
                  <span style={{ color: "var(--foreground)", fontWeight: 500 }}>click to browse</span>
                </p>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) processFile(file);
                  }}
                />
              </div>
            )}

            {/* Description */}
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe this image (e.g. chest X-ray showing..."
              rows={3}
              style={{
                width: "100%",
                resize: "none",
                fontSize: "13px",
                padding: "8px 10px",
                borderRadius: "8px",
                border: "0.5px solid var(--border)",
                background: "var(--input)",
                color: "var(--foreground)",
                fontFamily: "var(--font-body)",
                outline: "none",
                boxSizing: "border-box",
              }}
            />

            {/* Submitted count */}
            {images.length > 0 && (
              <p style={{ fontSize: "11px", color: "var(--muted-foreground)", margin: 0 }}>
                {images.length} image{images.length > 1 ? "s" : ""} already in agent state
                (ID{images.length > 1 ? "s" : ""} {images.map((i) => i.id).join(", ")})
              </p>
            )}

            {/* Submit */}
            <button
              onClick={handleSubmit}
              disabled={!base64 || !description.trim() || submitting}
              style={{
                width: "100%",
                padding: "9px",
                borderRadius: "8px",
                background: !base64 || !description.trim() ? "var(--muted)" : "var(--primary)",
                color: !base64 || !description.trim() ? "var(--muted-foreground)" : "var(--primary-foreground)",
                border: "none",
                cursor: !base64 || !description.trim() ? "not-allowed" : "pointer",
                fontSize: "13px",
                fontWeight: 500,
                transition: "all 0.15s ease",
              }}
            >
              {submitting ? "Adding..." : "Add to agent state"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}