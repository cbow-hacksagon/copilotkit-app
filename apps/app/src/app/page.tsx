"use client";
import { ExampleLayout } from "@/components/example-layout";
import { ExampleCanvas } from "@/components/example-canvas";
import { useGenerativeUIExamples, useExampleSuggestions } from "@/hooks";
import { CopilotChat } from "@copilotkit/react-core/v2";
import { ImageChatPopup } from "@/components/ImageChatPopup";
import { useAgent } from "@copilotkit/react-core/v2";
import { useState } from "react";

export default function HomePage() {
  useGenerativeUIExamples();
  const { agent } = useAgent();
  const [showDebug, setShowDebug] = useState(process.env.NEXT_PUBLIC_SHOW_DEBUG_PANEL === "true");

  const currentImages = agent.state?.Imaging || [];

  return (
    <div className="relative flex flex-col h-screen">
      <div className="flex-1 overflow-hidden">
        <CopilotChat
          input={{
            disclaimer: () => null,
            className: "pb-6",
	    
          }}
	  
        />
      </div>

      <ImageChatPopup />

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
                  ? currentImages.map((img) => `#${img.id} ${img.description}`).join("\n")
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
