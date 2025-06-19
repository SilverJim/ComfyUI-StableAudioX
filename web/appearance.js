import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-AudioX.appearance",
    async nodeCreated(node) {
        // AudioX nodes styling - Apply styling
        if (node.comfyClass === "AudioXModelLoader" ||
            node.comfyClass === "AudioXTextToAudio" ||
            node.comfyClass === "AudioXEnhancedTextToAudio" ||
            node.comfyClass === "AudioXTextToMusic" ||
            node.comfyClass === "AudioXEnhancedTextToMusic" ||
            node.comfyClass === "AudioXVideoToAudio" ||
            node.comfyClass === "AudioXEnhancedVideoToAudio" ||
            node.comfyClass === "AudioXVideoToMusic" ||
            node.comfyClass === "AudioXMultiModalGeneration" ||
            node.comfyClass === "AudioXAudioProcessor" ||
            node.comfyClass === "AudioXVolumeControl" ||
            node.comfyClass === "AudioXAdvancedVolumeControl" ||
            node.comfyClass === "AudioXVideoMuter" ||
            node.comfyClass === "AudioXVideoAudioCombiner" ||
            node.comfyClass === "AudioXPromptHelper") {
            node.color = "#ddaeff";
            node.bgcolor = "#a1cfa9";
        }
    }
});