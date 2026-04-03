"use client";
import { useState, useRef, useCallback } from "react";

interface UploadedImage {
  id: string;
  name: string;
  dataUrl: string;
  size: number;
}

export function ImageChatPopup() {
  const [open, setOpen] = useState(false);
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFiles = useCallback((files: FileList | null) => {
    if (!files) return;
    Array.from(files).forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        setImages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            name: file.name,
            dataUrl: e.target?.result as string,
            size: file.size,
          },
        ]);
      };
      reader.readAsDataURL(file);
    });
  }, []);

  const removeImage = (id: string) =>
    setImages((prev) => prev.filter((img) => img.id !== id));

  // expose images for your backend via window or a prop/callback
  // e.g. pass onImagesChange={(imgs) => sendToBackend(imgs)} as a prop

  return (
    <>
      {/* Trigger */}
      <button
        onClick={() => setOpen((p) => !p)}
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
            top: "-4px", right: "-4px",
            background: "#ef4444",
            color: "#fff",
            fontSize: "10px",
            width: "18px", height: "18px",
            borderRadius: "50%",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontWeight: 600,
          }}>
            {images.length}
          </span>
        )}
      </button>

      {/* Panel */}
      {open && (
        <div style={{
          position: "fixed",
          bottom: "84px", right: "24px",
          width: "300px",
          borderRadius: "12px",
          border: "0.5px solid var(--border)",
          background: "var(--card)",
          zIndex: 999,
          overflow: "hidden",
        }}>
          {/* Header */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "10px 14px",
            borderBottom: "0.5px solid var(--border)",
          }}>
            <span style={{ fontSize: "13px", fontWeight: 500, color: "var(--foreground)" }}>
              Images {images.length > 0 && `(${images.length})`}
            </span>
            <button
              onClick={() => setOpen(false)}
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted-foreground)", padding: "2px" }}
            >
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                <path d="M1 1l11 11M12 1L1 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </button>
          </div>

          {/* Drop zone */}
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); processFiles(e.dataTransfer.files); }}
            style={{
              margin: "12px",
              padding: "16px",
              border: `1.5px dashed ${dragging ? "var(--ring)" : "var(--border)"}`,
              borderRadius: "8px",
              textAlign: "center",
              cursor: "pointer",
              background: dragging ? "var(--accent)" : "transparent",
              transition: "all 0.15s ease",
            }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ margin: "0 auto 6px", display: "block", color: "var(--muted-foreground)" }}>
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <polyline points="17 8 12 3 7 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="12" y1="3" x2="12" y2="15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            <p style={{ fontSize: "12px", color: "var(--muted-foreground)", margin: 0 }}>
              Drop images or <span style={{ color: "var(--foreground)", fontWeight: 500 }}>click to browse</span>
            </p>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              multiple
              style={{ display: "none" }}
              onChange={(e) => processFiles(e.target.files)}
            />
          </div>

          {/* Thumbnails */}
          {images.length > 0 && (
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "8px",
              padding: "0 12px 12px",
            }}>
              {images.map((img) => (
                <div key={img.id} style={{ position: "relative", aspectRatio: "1" }}>
                  <img
                    src={img.dataUrl}
                    alt={img.name}
                    title={img.name}
                    style={{
                      width: "100%", height: "100%",
                      objectFit: "cover",
                      borderRadius: "6px",
                      border: "0.5px solid var(--border)",
                    }}
                  />
                  <button
                    onClick={() => removeImage(img.id)}
                    title="Remove"
                    style={{
                      position: "absolute", top: "-4px", right: "-4px",
                      width: "16px", height: "16px",
                      borderRadius: "50%",
                      background: "#ef4444",
                      border: "none", cursor: "pointer",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}
                  >
                    <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                      <path d="M1 1l6 6M7 1L1 7" stroke="#fff" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Clear all */}
          {images.length > 0 && (
            <div style={{ padding: "0 12px 12px" }}>
              <button
                onClick={() => setImages([])}
                style={{
                  width: "100%", padding: "6px",
                  fontSize: "12px", color: "var(--muted-foreground)",
                  background: "var(--muted)", border: "none",
                  borderRadius: "6px", cursor: "pointer",
                }}
              >
                Clear all
              </button>
            </div>
          )}
        </div>
      )}
    </>
  );
}