"use client";
import { ExampleLayout } from "@/components/example-layout";
import { ExampleCanvas } from "@/components/example-canvas";
import { useGenerativeUIExamples, useExampleSuggestions } from "@/hooks";
import { CopilotChat } from "@copilotkit/react-core/v2";
import {ImageChatPopup} from "@/components/ImageChatPopup";


export default function HomePage() {
  useGenerativeUIExamples();
  return (
  <>
        <CopilotChat
          input={{
            disclaimer: () => null,
            className: "pb-6",
	    
          }}
	  
        />

	<ImageChatPopup />
   </>
   );
}




  //return (
    //<ExampleLayout
     // chatContent={
       // <CopilotChat input={{ disclaimer: () => null, className: "pb-6" }} />
     // }
     // appContent={<ExampleCanvas />}
   // />
 // );}