"use client";
import { ExampleLayout } from "@/components/example-layout";
import { ExampleCanvas } from "@/components/example-canvas";
import { useGenerativeUIExamples, useExampleSuggestions } from "@/hooks";
import { CopilotChat } from "@copilotkit/react-core/v2";
export default function HomePage() {
  useGenerativeUIExamples();
  return (
        <CopilotChat
          input={{
            disclaimer: () => null,
            className: "pb-6",
          }}
        />
  );
}