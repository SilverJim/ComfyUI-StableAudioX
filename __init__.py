import os
import sys

# Safe import of ComfyUI folder_paths
try:
    import folder_paths as comfy_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    print("AudioX: ComfyUI folder_paths not available (running outside ComfyUI)")
    COMFYUI_AVAILABLE = False
    # Create a dummy folder_paths for testing
    class DummyFolderPaths:
        @staticmethod
        def get_filename_list(folder_type):
            return []
        @staticmethod
        def get_full_path(folder_type, filename):
            return None
    comfy_paths = DummyFolderPaths()

# Version identifier to force reload
__version__ = "1.0.9"

# Startup mode flag to prevent heavy operations during ComfyUI initialization
STARTUP_MODE = True

def set_runtime_mode():
    """Switch from startup mode to runtime mode."""
    global STARTUP_MODE
    STARTUP_MODE = False
    print("AudioX: Switched to runtime mode")

# Add the audiox directory to Python path
AUDIOX_ROOT = os.path.join(os.path.dirname(__file__), "audiox")
sys.path.insert(0, AUDIOX_ROOT)

# Debug environment information
print(f"AudioX: Python executable: {sys.executable}")
print(f"AudioX: Python version: {sys.version.split()[0]}")
print(f"AudioX: AudioX root: {AUDIOX_ROOT}")
print(f"AudioX: Current working directory: {os.getcwd()}")

# Ensure critical paths are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"AudioX: Added to path: {current_dir}")

# Ensure all required dependencies are available
def ensure_dependencies():
    """Ensure all AudioX dependencies are available."""
    # Complete list of required dependencies
    required_deps = [
        "descript-audio-codec",
        "einops-exts",
        "x-transformers",
        "alias-free-torch",
        "vector-quantize-pytorch",
        "local-attention",
        "k-diffusion",
        "aeiou",
        "auraloss",
        "encodec",
        "laion-clap",
        "prefigure",
        "v-diffusion-pytorch"
    ]

    missing_deps = []

    # Check all dependencies
    dependency_checks = [
        ("dac", "descript-audio-codec"),
        ("einops_exts", "einops-exts"),
        ("x_transformers", "x-transformers"),
        ("alias_free_torch", "alias-free-torch"),
        ("vector_quantize_pytorch", "vector-quantize-pytorch"),
        ("local_attention", "local-attention"),
        ("k_diffusion", "k-diffusion"),
        ("aeiou", "aeiou"),
        ("auraloss", "auraloss"),
        ("encodec", "encodec"),
        ("laion_clap", "laion-clap"),
        ("prefigure", "prefigure"),
        ("diffusion", "v-diffusion-pytorch")
    ]

    for import_name, package_name in dependency_checks:
        try:
            __import__(import_name)
            print(f"AudioX: {package_name} available")
        except ImportError as e:
            print(f"AudioX: {package_name} not available: {e}")
            missing_deps.append(package_name)

    # Install missing dependencies immediately
    if missing_deps:
        print(f"AudioX: Installing {len(missing_deps)} missing dependencies...")
        print(f"AudioX: Using Python: {sys.executable}")

        try:
            import subprocess

            # Install all missing dependencies in one command for efficiency
            install_cmd = [sys.executable, "-m", "pip", "install"] + missing_deps
            print(f"AudioX: Running: {' '.join(install_cmd)}")

            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("AudioX: All dependencies installed successfully!")
                print("AudioX: Reloading modules...")

                # Clear import cache to force reload
                import importlib
                modules_to_reload = [
                    'dac', 'einops_exts', 'x_transformers', 'alias_free_torch',
                    'vector_quantize_pytorch', 'local_attention', 'k_diffusion',
                    'aeiou', 'auraloss', 'encodec', 'laion_clap', 'prefigure',
                    'diffusion'
                ]

                for module_name in modules_to_reload:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])

                # Re-check critical dependencies
                missing_deps.clear()
                for import_name, package_name in dependency_checks:
                    try:
                        __import__(import_name)
                        print(f"AudioX: ✓ {package_name} now available")
                    except ImportError:
                        missing_deps.append(package_name)
                        print(f"AudioX: ✗ {package_name} still missing")

            else:
                print(f"AudioX: Installation failed: {result.stderr}")
                print("AudioX: Trying individual installations...")

                # Try installing each dependency individually
                for dep in missing_deps[:]:
                    try:
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", dep
                        ], capture_output=True, text=True, timeout=120)

                        if result.returncode == 0:
                            print(f"AudioX: ✓ {dep} installed")
                            missing_deps.remove(dep)
                        else:
                            print(f"AudioX: ✗ Failed to install {dep}")
                    except Exception as e:
                        print(f"AudioX: Error installing {dep}: {e}")

        except Exception as install_error:
            print(f"AudioX: Installation error: {install_error}")

    if missing_deps:
        print(f"AudioX: ⚠️  Still missing: {missing_deps}")
        print("AudioX: Some features may not work correctly")
    else:
        print("AudioX: ✅ All dependencies available!")

    return len(missing_deps) == 0

# EMERGENCY: Force install critical dependencies immediately
print("AudioX: EMERGENCY DEPENDENCY CHECK...")
try:
    import vector_quantize_pytorch
    print("AudioX: vector_quantize_pytorch already available")
except ImportError:
    print("AudioX: INSTALLING vector_quantize_pytorch NOW...")
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "vector-quantize-pytorch"
        ], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("AudioX: ✅ vector_quantize_pytorch installed successfully!")
            # Force reload the module
            try:
                import importlib
                import vector_quantize_pytorch
                importlib.reload(vector_quantize_pytorch)
            except:
                pass
        else:
            print(f"AudioX: ❌ Failed to install vector_quantize_pytorch: {result.stderr}")
    except Exception as e:
        print(f"AudioX: ❌ Installation error: {e}")

# Quick dependency check without heavy operations during startup
print("AudioX: EMERGENCY DEPENDENCY CHECK...")
try:
    # Only check critical dependencies that are fast to import
    import dac
    import einops_exts
    print("AudioX: Critical dependencies available")
    deps_ok = True
except ImportError as e:
    print(f"AudioX: Critical dependency missing: {e}")
    print("AudioX: Will attempt full dependency check later...")
    deps_ok = ensure_dependencies()

# Import our nodes with error handling and lazy loading
try:
    print("AudioX: Importing core nodes...")

    # Import nodes with timeout protection
    import concurrent.futures
    import sys

    def import_core_nodes():
        """Import core nodes in a separate thread."""
        from .nodes import (
            AudioXModelLoader,
            AudioXTextToAudio,
            AudioXEnhancedTextToAudio,
            AudioXTextToMusic,
            AudioXEnhancedTextToMusic,
            AudioXVideoToAudio,
            AudioXEnhancedVideoToAudio,
            AudioXVideoToMusic,
            AudioXMultiModalGeneration,
            AudioXAudioProcessor,
            AudioXVolumeControl,
            AudioXAdvancedVolumeControl,
            AudioXVideoMuter,
            AudioXVideoAudioCombiner,
            AudioXPromptHelper
        )
        return (
            AudioXModelLoader, AudioXTextToAudio, AudioXEnhancedTextToAudio,
            AudioXTextToMusic, AudioXEnhancedTextToMusic, AudioXVideoToAudio,
            AudioXEnhancedVideoToAudio, AudioXVideoToMusic, AudioXMultiModalGeneration,
            AudioXAudioProcessor, AudioXVolumeControl, AudioXAdvancedVolumeControl,
            AudioXVideoMuter, AudioXVideoAudioCombiner, AudioXPromptHelper
        )

    # Use ThreadPoolExecutor with timeout for import
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(import_core_nodes)
        try:
            # 30 second timeout for core node import
            core_nodes = future.result(timeout=30)
            (AudioXModelLoader, AudioXTextToAudio, AudioXEnhancedTextToAudio,
             AudioXTextToMusic, AudioXEnhancedTextToMusic, AudioXVideoToAudio,
             AudioXEnhancedVideoToAudio, AudioXVideoToMusic, AudioXMultiModalGeneration,
             AudioXAudioProcessor, AudioXVolumeControl, AudioXAdvancedVolumeControl,
             AudioXVideoMuter, AudioXVideoAudioCombiner, AudioXPromptHelper) = core_nodes
            print("AudioX: ✅ Core nodes imported successfully!")
        except concurrent.futures.TimeoutError:
            print("AudioX: ⚠️ Core node import timed out, using placeholder nodes")
            raise ImportError("Core node import timeout")



except Exception as e:
    error_message = str(e)
    print(f"AudioX: ❌ Failed to import core nodes: {error_message}")
    print("AudioX: This might be due to missing dependencies or network issues.")
    print("AudioX: Creating placeholder nodes...")

    # Create placeholder nodes that will show error messages
    class PlaceholderNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error_info": ("STRING", {"default": f"AudioX import failed: {error_message}"})}}

        RETURN_TYPES = ("STRING",)
        FUNCTION = "show_error"
        CATEGORY = "AudioX/Error"

        def show_error(self, error_info):
            raise RuntimeError(f"AudioX nodes are not available: {error_info}")

    # Use placeholder for all nodes
    AudioXModelLoader = PlaceholderNode
    AudioXTextToAudio = PlaceholderNode
    AudioXEnhancedTextToAudio = PlaceholderNode
    AudioXTextToMusic = PlaceholderNode
    AudioXEnhancedTextToMusic = PlaceholderNode
    AudioXVideoToAudio = PlaceholderNode
    AudioXEnhancedVideoToAudio = PlaceholderNode
    AudioXVideoToMusic = PlaceholderNode
    AudioXMultiModalGeneration = PlaceholderNode
    AudioXAudioProcessor = PlaceholderNode
    AudioXVolumeControl = PlaceholderNode
    AudioXAdvancedVolumeControl = PlaceholderNode
    AudioXVideoMuter = PlaceholderNode
    AudioXVideoAudioCombiner = PlaceholderNode
    AudioXPromptHelper = PlaceholderNode


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AudioXModelLoader": AudioXModelLoader,
    "AudioXTextToAudio": AudioXTextToAudio,
    "AudioXEnhancedTextToAudio": AudioXEnhancedTextToAudio,
    "AudioXTextToMusic": AudioXTextToMusic,
    "AudioXEnhancedTextToMusic": AudioXEnhancedTextToMusic,
    "AudioXVideoToAudio": AudioXVideoToAudio,
    "AudioXEnhancedVideoToAudio": AudioXEnhancedVideoToAudio,
    "AudioXVideoToMusic": AudioXVideoToMusic,
    "AudioXMultiModalGeneration": AudioXMultiModalGeneration,
    "AudioXAudioProcessor": AudioXAudioProcessor,
    "AudioXVolumeControl": AudioXVolumeControl,
    "AudioXAdvancedVolumeControl": AudioXAdvancedVolumeControl,
    "AudioXVideoMuter": AudioXVideoMuter,
    "AudioXVideoAudioCombiner": AudioXVideoAudioCombiner,
    "AudioXPromptHelper": AudioXPromptHelper,

}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioXModelLoader": "AudioX Model Loader",
    "AudioXTextToAudio": "AudioX Text to Audio",
    "AudioXEnhancedTextToAudio": "AudioX Enhanced Text to Audio",
    "AudioXTextToMusic": "AudioX Text to Music",
    "AudioXEnhancedTextToMusic": "AudioX Enhanced Text to Music",
    "AudioXVideoToAudio": "AudioX Video to Audio",
    "AudioXEnhancedVideoToAudio": "AudioX Enhanced Video to Audio",
    "AudioXVideoToMusic": "AudioX Video to Music",
    "AudioXMultiModalGeneration": "AudioX Multi-Modal Generation",
    "AudioXAudioProcessor": "AudioX Audio Processor",
    "AudioXVolumeControl": "AudioX Volume Control",
    "AudioXAdvancedVolumeControl": "AudioX Advanced Volume Control",
    "AudioXVideoMuter": "AudioX Video Muter",
    "AudioXVideoAudioCombiner": "AudioX Video Audio Combiner",
    "AudioXPromptHelper": "AudioX Prompt Helper",
}



# Web directory for any web components
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
