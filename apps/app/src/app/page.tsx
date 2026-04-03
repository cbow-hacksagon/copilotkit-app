"use client";
import { ExampleLayout } from "@/components/example-layout";
import { ExampleCanvas } from "@/components/example-canvas";
import { useGenerativeUIExamples, useExampleSuggestions } from "@/hooks";
import { CopilotChat } from "@copilotkit/react-core/v2";
import { useAgent } from "@copilotkit/react-core/v2";
import { useRef, useState, useEffect } from "react";

interface ImageEntry {
  id: string;
  filename: string;
  data_url: string;
  description: string;
  timestamp: string;
}

export default function HomePage() {
  useGenerativeUIExamples();
  const { agent } = useAgent();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const descriptionRef = useRef<HTMLInputElement>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [showPanel, setShowPanel] = useState(false);
  const [showDebug, setShowDebug] = useState(process.env.NEXT_PUBLIC_SHOW_DEBUG_PANEL === "true");
  const [imageCount, setImageCount] = useState(0);

  const currentImages: ImageEntry[] = agent.state?.Imaging || [];

  useEffect(() => {
    setImageCount(currentImages.length);
  }, [currentImages.length]);

  const handleAddFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    const invalidFiles = fileArray.filter(
      (f) => !f.type.startsWith("image/") || f.size > 5242880
    );

    if (invalidFiles.length > 0) {
      const names = invalidFiles.map((f) => f.name).join(", ");
      setUploadError(`Invalid files: ${names}. Only images under 5MB are accepted.`);
      return;
    }

    setUploadError(null);
    setPendingFiles(fileArray);
    if (descriptionRef.current) descriptionRef.current.value = "";
    setShowPanel(true);
  };

  const handleUploadConfirm = async () => {
    if (pendingFiles.length === 0) return;

    try {
      setUploading(true);
      setUploadError(null);

      const description = descriptionRef.current?.value || "";
      const existingImages: ImageEntry[] = agent.state?.Imaging || [];
      const newImages: ImageEntry[] = [...existingImages];

      for (const file of pendingFiles) {
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
          reader.readAsDataURL(file);
        });

        newImages.push({
          id: crypto.randomUUID(),
          filename: file.name,
          data_url: dataUrl,
          description,
          timestamp: new Date().toISOString(),
        });
      }

      agent.setState({ Imaging: newImages });
      setPendingFiles([]);
      setShowPanel(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown upload error";
      setUploadError(message);
    } finally {
      setUploading(false);
    }
  };

  const handleUploadCancel = () => {
    setPendingFiles([]);
    setShowPanel(false);
    setUploadError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeImage = (id: string) => {
    const currentImages: ImageEntry[] = agent.state?.Imaging || [];
    const updated = currentImages.filter((img) => img.id !== id);
    agent.setState({ Imaging: updated });
  };

  return (
    <div className="relative flex flex-col h-screen">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={handleFileChange}
      />

      <div className="flex-1 overflow-hidden">
        <CopilotChat
          input={{
            disclaimer: () => null,
            className: "pb-6",
          }}
        />
      </div>

      {/* Debug toggle FAB - bottom-left */}
      {process.env.NEXT_PUBLIC_SHOW_DEBUG_PANEL === "true" && (
        <button
          onClick={() => setShowDebug(!showDebug)}
          className="fixed bottom-4 left-4 z-40 w-10 h-10 rounded-full bg-[#f8f8f8] border shadow-sm flex items-center justify-center text-[#5d5d5d] hover:bg-[#e8e8e8] transition-colors text-xs font-mono"
          title="Toggle debug panel"
        >
          {showDebug ? "✕" : "🐛"}
        </button>
      )}

      {/* Image upload FAB - bottom-right */}
      <button
        onClick={handleAddFile}
        disabled={uploading}
        className="fixed bottom-4 right-4 z-40 w-12 h-12 rounded-full bg-black text-white shadow-lg flex items-center justify-center hover:opacity-70 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        title={uploading ? "Uploading..." : "Attach medical image"}
      >
        {uploading ? (
          <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        )}
        {imageCount > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-red-500 text-white text-[10px] font-bold flex items-center justify-center">
            {imageCount}
          </span>
        )}
      </button>

      {/* Image upload popup panel */}
      {showPanel && (
        <div className="fixed inset-0 z-50 flex items-end justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/30"
            onClick={handleUploadCancel}
          />

          {/* Panel */}
          <div className="relative w-full max-w-md bg-white rounded-t-2xl shadow-2xl border-b-0 p-5 mb-0 animate-slide-up">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-[#1d1d1d]">
                Attach Image{pendingFiles.length > 0 ? `s (${pendingFiles.length})` : ""}
              </h3>
              <button
                onClick={handleUploadCancel}
                className="w-8 h-8 rounded-full flex items-center justify-center text-[#5d5d5d] hover:bg-[#f8f8f8] transition-colors"
              >
                ×
              </button>
            </div>

            {pendingFiles.length > 0 && (
              <div className="mb-4">
                <p className="text-xs text-[#5d5d5d] mb-2">Selected files:</p>
                <div className="flex flex-wrap gap-1.5">
                  {pendingFiles.map((f, i) => (
                    <span
                      key={i}
                      className="px-2.5 py-1 bg-[#f8f8f8] rounded-full text-xs text-[#333] font-medium truncate max-w-[180px]"
                    >
                      {f.name}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="mb-4">
              <label className="block text-xs text-[#5d5d5d] mb-1.5">
                Description <span className="text-[#999]">(optional)</span>
              </label>
              <input
                ref={descriptionRef}
                type="text"
                placeholder="e.g., X-Ray of chest, MRI brain scan..."
                className="w-full px-3 py-2 border border-[#e0e0e0] rounded-lg text-sm text-[#1d1d1d] placeholder:text-[#999] focus:outline-none focus:border-black focus:ring-1 focus:ring-black transition-colors"
              />
            </div>

            {uploadError && (
              <div className="mb-4 bg-red-50 border border-red-200 p-2.5 rounded-lg text-red-700 text-xs">
                {uploadError}
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={handleUploadConfirm}
                disabled={uploading}
                className="flex-1 py-2.5 bg-black text-white rounded-lg text-sm font-medium hover:opacity-80 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploading ? "Uploading..." : "Upload to Imaging"}
              </button>
              <button
                onClick={handleUploadCancel}
                className="px-5 py-2.5 bg-[#f8f8f8] text-[#333] rounded-lg text-sm font-medium hover:bg-[#e8e8e8] transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Uploaded images indicator - shown above FAB area when panel is closed */}
      {!showPanel && currentImages.length > 0 && (
        <div className="fixed bottom-20 right-4 z-30 max-w-[240px]">
          <div className="bg-white rounded-lg shadow-md border p-2">
            <p className="text-[10px] font-semibold text-[#5d5d5d] mb-1.5 px-1">
              Attached Images
            </p>
            <div className="flex flex-col gap-1 max-h-32 overflow-y-auto">
              {currentImages.map((img) => (
                <div
                  key={img.id}
                  className="flex items-center gap-1.5 px-2 py-1 bg-[#f8f8f8] rounded text-[11px]"
                >
                  <span className="text-[#333] font-medium truncate flex-1">
                    {img.filename}
                  </span>
                  {img.description && (
                    <span className="text-[#999] truncate max-w-[80px]">
                      {img.description}
                    </span>
                  )}
                  <button
                    onClick={() => removeImage(img.id)}
                    className="text-red-400 hover:text-red-600 font-bold shrink-0"
                    title="Remove"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Debug panel */}
      {showDebug && (
        <div className="fixed top-0 right-0 w-80 h-full bg-[#1a1a2e] text-[#00ff88] font-mono text-[11px] overflow-auto p-4 z-50 shadow-2xl">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-bold text-white text-sm">Agent State Debug</h3>
            <button
              onClick={() => setShowDebug(false)}
              className="text-gray-400 hover:text-white w-7 h-7 rounded-full flex items-center justify-center hover:bg-white/10"
            >
              ×
            </button>
          </div>

          <div className="space-y-3">
            <div>
              <p className="text-yellow-400 font-bold mb-1 text-xs">Imaging ({currentImages.length} images):</p>
              <pre className="bg-black/40 p-2 rounded text-[10px] overflow-x-auto whitespace-pre-wrap break-all max-h-40">
                {currentImages.length > 0
                  ? currentImages.map((img) => `${img.filename}${img.description ? ` — "${img.description}"` : ""}`).join("\n")
                  : "(empty)"}
              </pre>
            </div>

            <div>
              <p className="text-yellow-400 font-bold mb-1 text-xs">Full State Keys:</p>
              <pre className="bg-black/40 p-2 rounded text-[10px] overflow-x-auto">
                {agent.state ? Object.keys(agent.state).join(", ") : "(none)"}
              </pre>
            </div>

            <div>
              <p className="text-yellow-400 font-bold mb-1 text-xs">Agent Running:</p>
              <pre className="bg-black/40 p-2 rounded text-[10px]">
                {String(agent.isRunning)}
              </pre>
            </div>

            <div>
              <p className="text-yellow-400 font-bold mb-1 text-xs">Raw Imaging JSON:</p>
              <pre className="bg-black/40 p-2 rounded text-[10px] overflow-x-auto whitespace-pre-wrap break-all max-h-48">
                {JSON.stringify(currentImages, null, 2) || "(empty)"}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
