import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import os
import copy

# Safe import of ComfyUI folder_paths
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    print("AudioX: ComfyUI folder_paths not available (running outside ComfyUI)")
    COMFYUI_AVAILABLE = False
    # Create a dummy folder_paths for testing
    class DummyFolderPaths:
        @staticmethod
        def get_filename_list(folder_type):
            return ["dummy_model.safetensors"]
        @staticmethod
        def get_full_path(folder_type, filename):
            return f"/dummy/path/{filename}"
    folder_paths = DummyFolderPaths()

from .audiox_utils import (
    load_audiox_model_from_file,
    prepare_text_conditioning,
    prepare_video_conditioning,
    prepare_audio_conditioning,
    convert_image_to_video,
    postprocess_audio_output,
    validate_inputs,
    get_generation_parameters,
    get_device,
    clear_model_cache,
    create_empty_conditioning_values,
    create_minimal_conditioning,
    create_video_only_conditioning,
    create_enhanced_video_conditioning,
    enhance_audio_prompt,
    get_audio_prompt_templates,
    create_safe_negative_conditioning,
    analyze_prompt_specificity,
    calculate_adaptive_cfg,
    expand_audio_keywords,
    emphasize_key_terms,
    create_multi_aspect_conditioning,
    create_super_enhanced_prompt
)

# Global variable to cache the generation function
_generate_diffusion_cond = None
_import_attempted = False
_import_error = None

def safe_import_audiox():
    """Safely import AudioX generation functions with proper error handling and Windows-compatible timeout."""
    global _generate_diffusion_cond, _import_attempted, _import_error

    # Check if we're in startup mode and defer import if so
    try:
        from . import STARTUP_MODE
        if STARTUP_MODE:
            print("AudioX: Deferring generation function import (startup mode)")
            # Return a placeholder that will trigger actual import when called
            def deferred_import(*args, **kwargs):
                # Switch to runtime mode and import for real
                from . import set_runtime_mode
                set_runtime_mode()
                return safe_import_audiox()(*args, **kwargs)
            return deferred_import
    except ImportError:
        pass  # Continue with normal import if startup mode not available

    # Return cached result if already attempted
    if _import_attempted:
        if _generate_diffusion_cond is not None:
            return _generate_diffusion_cond
        else:
            raise ImportError(f"AudioX generation functions not available: {_import_error}")

    try:
        print("AudioX: Importing generation functions...")
        import concurrent.futures
        import sys

        def import_generation_function():
            """Import function to run in separate thread with timeout."""
            try:
                from stable_audio_tools.inference.generation import generate_diffusion_cond
                return generate_diffusion_cond
            except Exception as e:
                raise ImportError(f"Failed to import generation functions: {e}")

        # Use ThreadPoolExecutor with timeout for Windows compatibility
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(import_generation_function)
            try:
                # 60 second timeout for import (increased from 30 for stability)
                result = future.result(timeout=60)
                _generate_diffusion_cond = result
                _import_attempted = True
                print("AudioX: ✅ Generation functions imported successfully!")
                return result
            except concurrent.futures.TimeoutError:
                _import_error = "Import timeout - this may be due to network issues downloading models"
                _import_attempted = True
                raise TimeoutError(_import_error)
            except Exception as e:
                _import_error = str(e)
                _import_attempted = True
                raise

    except Exception as e:
        _import_error = str(e)
        _import_attempted = True
        print(f"AudioX: ❌ Failed to import generation functions: {e}")
        raise ImportError(
            f"AudioX dependencies not found or import timed out. "
            f"This may be due to network issues downloading transformer models. "
            f"Error: {e}"
        )

class AudioXModelLoader:
    """Node for loading AudioX models from ComfyUI model directories."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select AudioX model file from models/diffusion_models/"
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "config_name": (["auto"] + folder_paths.get_filename_list("configs"), {
                    "default": "auto",
                    "tooltip": "Optional: Select config file from models/configs/ (auto-detect if available)"
                }),
            }
        }

    RETURN_TYPES = ("AUDIOX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AudioX/Models"

    # Add this to help with debugging
    DESCRIPTION = "Load AudioX models from ComfyUI model directories. Fixed version 1.0.1"

    def load_model(self, model_name: str, device: str, config_name: str = "auto"):
        """Load the AudioX model from ComfyUI model directories."""
        try:
            if device == "auto":
                device = get_device()

            print(f"AudioX: Loading model '{model_name}' on device '{device}'")

            # Get full path to model file
            model_path = folder_paths.get_full_path("diffusion_models", model_name)
            if not model_path:
                available_models = folder_paths.get_filename_list("diffusion_models")
                raise FileNotFoundError(
                    f"Model file not found: {model_name}\n"
                    f"Available models: {available_models}\n"
                    f"Please place AudioX model files in: models/diffusion_models/"
                )

            print(f"AudioX: Model path: {model_path}")

            # Check if model file exists and is readable
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file does not exist: {model_path}")

            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"AudioX: Model file size: {file_size:.1f} MB")

            # Handle config file
            config_path = None
            if config_name != "auto":
                config_path = folder_paths.get_full_path("configs", config_name)
                if config_path:
                    print(f"AudioX: Using config: {config_path}")
                else:
                    print(f"AudioX: Config file not found: {config_name}")
            else:
                # Try to auto-detect config file
                config_path = self._find_config_file(model_path, model_name)
                if config_path:
                    print(f"AudioX: Auto-detected config: {config_path}")
                else:
                    print("AudioX: No config file found, using default")

            # Load the model
            print("AudioX: Starting model loading...")
            model, model_config = load_audiox_model_from_file(model_path, config_path, device)
            print("AudioX: Model loaded successfully!")

            return ({
                "model": model,
                "config": model_config,
                "device": device,
                "model_path": model_path,
                "config_path": config_path
            },)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"AudioX Model Loading Error: {error_details}")
            raise RuntimeError(f"Failed to load AudioX model: {str(e)}")

    def _find_config_file(self, model_path: str, model_name: str) -> Optional[str]:
        """Try to find a matching config file for the model."""
        import os

        # Try same directory as model first
        model_dir = os.path.dirname(model_path)
        model_base = os.path.splitext(model_name)[0]

        # Common config file patterns
        config_patterns = [
            f"{model_base}.json",
            f"{model_base}_config.json",
            "config.json",
            "model_config.json",
            "audiox_config.json"  # Added AudioX specific pattern
        ]

        # Check model directory first
        for pattern in config_patterns:
            config_path = os.path.join(model_dir, pattern)
            if os.path.exists(config_path):
                print(f"AudioX: Found config in model directory: {config_path}")
                return config_path

        # Try configs directory
        for pattern in config_patterns:
            config_path = folder_paths.get_full_path("configs", pattern)
            if config_path and os.path.exists(config_path):
                print(f"AudioX: Found config in configs directory: {config_path}")
                return config_path

        print("AudioX: No config file found, will use default")
        return None

class AudioXTextToAudio:
    """Node for text-to-audio generation using AudioX."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "Typing on a keyboard",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "AudioX/Generation"
    
    def generate_audio(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float,
                      seed: int, duration_seconds: float):
        """Generate audio from text prompt."""
        try:
            # Import generation function only when actually needed
            print("AudioX: Loading generation function for audio generation...")
            generate_diffusion_cond = safe_import_audiox()
            
            # Enhance the text prompt for better audio generation
            enhanced_text_prompt = enhance_audio_prompt(text_prompt)
            print(f"AudioX: Enhanced text prompt: '{enhanced_text_prompt}'")

            # Validate inputs
            validate_inputs(text_prompt=enhanced_text_prompt)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # Try minimal conditioning first (text-only) with enhanced prompt
            conditioning = [create_minimal_conditioning(enhanced_text_prompt, int(duration_seconds), device)]
            
            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )
            
            # Generate audio with fallback to full conditioning if needed
            with torch.no_grad():
                try:
                    output = generate_diffusion_cond(
                        copied_model,
                        conditioning=conditioning,
                        device=device,
                        **gen_params
                    )
                except (ValueError, RuntimeError) as e:
                    if "not found in batch metadata" in str(e) or "size of tensor" in str(e):
                        print(f"AudioX: Minimal conditioning failed ({e}), trying full conditioning...")
                        # Fall back to full conditioning with empty values using enhanced prompt
                        empty_values = create_empty_conditioning_values(device, int(duration_seconds))
                        conditioning = [{
                            "text_prompt": enhanced_text_prompt,
                            "video_prompt": empty_values["video_prompt"],  # Empty video tensor
                            "audio_prompt": empty_values["audio_prompt"],  # Empty audio tensor
                            "seconds_start": 0,
                            "seconds_total": int(duration_seconds)
                        }]
                        output = generate_diffusion_cond(
                            copied_model, # Use copied model in fallback too
                            conditioning=conditioning,
                            device=device,
                            **gen_params
                        )
                    else:
                        raise
            
            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)
            
        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {str(e)}")

class AudioXTextToMusic:
    """Node for text-to-music generation using AudioX."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "A music with piano and violin",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_music"
    CATEGORY = "AudioX/Generation"
    
    def generate_music(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float, 
                      seed: int, duration_seconds: float):
        """Generate music from text prompt."""
        try:
            generate_diffusion_cond = safe_import_audiox()
            
            # Enhance the text prompt for better music generation
            enhanced_text_prompt = enhance_audio_prompt(text_prompt)
            print(f"AudioX: Enhanced music prompt: '{enhanced_text_prompt}'")

            # Validate inputs
            validate_inputs(text_prompt=enhanced_text_prompt)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # Try minimal conditioning first (text-only) with enhanced prompt
            conditioning = [create_minimal_conditioning(enhanced_text_prompt, int(duration_seconds), device)]
            
            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )
            
            # Generate music with fallback to full conditioning if needed
            with torch.no_grad():
                try:
                    output = generate_diffusion_cond(
                        copied_model,
                        conditioning=conditioning,
                        device=device,
                        **gen_params
                    )
                except (ValueError, RuntimeError) as e:
                    if "not found in batch metadata" in str(e) or "size of tensor" in str(e):
                        print(f"AudioX: Minimal conditioning failed ({e}), trying full conditioning...")
                        # Fall back to full conditioning with empty values using enhanced prompt
                        empty_values = create_empty_conditioning_values(device, int(duration_seconds))
                        conditioning = [{
                            "text_prompt": enhanced_text_prompt,
                            "video_prompt": empty_values["video_prompt"],  # Empty video tensor
                            "audio_prompt": empty_values["audio_prompt"],  # Empty audio tensor
                            "seconds_start": 0,
                            "seconds_total": int(duration_seconds)
                        }]
                        output = generate_diffusion_cond(
                            copied_model, # Use copied model in fallback too
                            conditioning=conditioning,
                            device=device,
                            **gen_params
                        )
                    else:
                        raise
            
            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)
            
        except Exception as e:
            raise RuntimeError(f"Music generation failed: {str(e)}")

class AudioXVideoToAudio:
    """Node for video-to-audio generation using AudioX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "video": ("IMAGE",),  # ComfyUI video format
                "text_prompt": ("STRING", {
                    "default": "Generate realistic audio that matches the visual content and actions in this video",
                    "multiline": True,
                    "tooltip": "Describe the audio you want to generate for this video"
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio_from_video"
    CATEGORY = "AudioX/Generation"

    def generate_audio_from_video(self, model: Dict, video: torch.Tensor, text_prompt: str,
                                 steps: int, cfg_scale: float, seed: int, duration_seconds: float):
        """Generate audio from video input."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Enhance the text prompt for better audio generation
            enhanced_text_prompt = enhance_audio_prompt(text_prompt)
            print(f"AudioX: Enhanced prompt: '{enhanced_text_prompt}'")

            # Validate inputs
            validate_inputs(text_prompt=enhanced_text_prompt, video_tensor=video)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # CRITICAL FIX: Use specialized video-only conditioning to avoid sequence length conflicts
            print(f"AudioX: Creating video-only conditioning for {duration_seconds}s generation")
            conditioning = create_video_only_conditioning(video, enhanced_text_prompt, duration_seconds, device)

            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate audio
            with torch.no_grad():
                output = generate_diffusion_cond(
                    copied_model,
                    conditioning=[conditioning],
                    device=device,
                    **gen_params
                )

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Video-to-audio generation failed: {str(e)}")

class AudioXEnhancedVideoToAudio:
    """Enhanced node for video-to-audio generation with advanced conditioning controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "video": ("IMAGE",),  # ComfyUI video format
                "text_prompt": ("STRING", {
                    "default": "Generate realistic audio that matches the visual content",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "text_cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for text conditioning"
                }),
                "video_cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for video conditioning"
                }),
                "text_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Weight for text conditioning influence"
                }),
                "video_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Weight for video conditioning influence"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Negative text prompt to avoid certain audio characteristics"
                }),
                "prompt_template": (["none", "music_ambient", "music_upbeat", "nature_forest", "nature_ocean", "urban_traffic", "action_footsteps"], {
                    "default": "none",
                    "tooltip": "Use predefined prompt template"
                }),
                "enhance_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically enhance prompt with audio-specific keywords"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_enhanced_audio_from_video"
    CATEGORY = "AudioX/Generation"

    def generate_enhanced_audio_from_video(self, model: Dict, video: torch.Tensor, text_prompt: str,
                                         steps: int, text_cfg_scale: float, video_cfg_scale: float,
                                         text_weight: float, video_weight: float, seed: int, duration_seconds: float,
                                         negative_prompt: str = "", prompt_template: str = "none", enhance_prompt: bool = True):
        """Generate audio from video with enhanced conditioning controls."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Apply prompt template if selected
            if prompt_template != "none":
                templates = get_audio_prompt_templates()
                template_parts = prompt_template.split('_')
                if len(template_parts) == 2:
                    category, subcategory = template_parts
                    if category in templates and subcategory in templates[category]:
                        template_text = templates[category][subcategory]
                        text_prompt = f"{text_prompt}, {template_text}" if text_prompt else template_text

            # Enhance prompt if requested
            if enhance_prompt:
                text_prompt = enhance_audio_prompt(text_prompt)

            # Validate inputs
            validate_inputs(text_prompt=text_prompt, video_tensor=video)

            # Get model components
            audiox_model = model["model"]
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # Create enhanced conditioning
            print(f"AudioX Enhanced: Creating conditioning with text_weight={text_weight}, video_weight={video_weight}")
            conditioning = create_enhanced_video_conditioning(
                video, text_prompt, duration_seconds, device,
                text_weight=text_weight, video_weight=video_weight,
                negative_prompt=negative_prompt
            )

            # Get generation parameters with weighted CFG
            # Use average of text and video CFG scales weighted by their respective weights
            total_weight = text_weight + video_weight
            if total_weight > 0:
                effective_cfg_scale = (text_cfg_scale * text_weight + video_cfg_scale * video_weight) / total_weight
            else:
                effective_cfg_scale = (text_cfg_scale + video_cfg_scale) / 2

            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=effective_cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            print(f"AudioX Enhanced: Using effective CFG scale: {effective_cfg_scale:.2f}")

            # Generate audio with enhanced conditioning
            with torch.no_grad():
                output = generate_diffusion_cond(
                    copied_model,
                    conditioning=[conditioning],
                    device=device,
                    **gen_params
                )

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Enhanced video-to-audio generation failed: {str(e)}")

class AudioXEnhancedTextToAudio:
    """Enhanced node for text-to-audio generation with advanced prompt controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "Typing on a keyboard",
                    "multiline": True,
                    "tooltip": "Describe the audio you want to generate"
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale for prompt adherence"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "muffled, distorted, low quality, noise, silence",
                    "multiline": True,
                    "tooltip": "Negative text prompt (currently logged only - implementation pending)"
                }),
                "prompt_template": (["none", "music_ambient", "music_upbeat", "nature_forest", "nature_ocean", "urban_traffic", "action_footsteps", "action_impact", "action_mechanical"], {
                    "default": "none",
                    "tooltip": "Use predefined prompt template"
                }),
                "enhance_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically enhance prompt with audio-specific keywords"
                }),
                "style_modifier": (["none", "cinematic", "realistic", "ambient", "dramatic", "peaceful", "energetic"], {
                    "default": "none",
                    "tooltip": "Add style modifier to the prompt"
                }),
                "conditioning_mode": (["standard", "enhanced", "super_enhanced", "multi_aspect"], {
                    "default": "enhanced",
                    "tooltip": "Conditioning enhancement level"
                }),
                "adaptive_cfg": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically adjust CFG based on prompt specificity"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_enhanced_audio"
    CATEGORY = "AudioX/Generation"

    def generate_enhanced_audio(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float,
                              seed: int, duration_seconds: float, negative_prompt: str = "",
                              prompt_template: str = "none", enhance_prompt: bool = True,
                              style_modifier: str = "none", conditioning_mode: str = "enhanced",
                              adaptive_cfg: bool = True):
        """Generate audio from text prompt with enhanced controls."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Apply prompt template if selected
            if prompt_template != "none":
                templates = get_audio_prompt_templates()
                template_parts = prompt_template.split('_')
                if len(template_parts) == 2:
                    category, subcategory = template_parts
                    if category in templates and subcategory in templates[category]:
                        template_text = templates[category][subcategory]
                        text_prompt = f"{text_prompt}, {template_text}" if text_prompt else template_text

            # Apply advanced conditioning based on mode
            if conditioning_mode == "super_enhanced":
                text_prompt = create_super_enhanced_prompt(text_prompt, "audio", style_modifier, True)
            else:
                # Add style modifier
                if style_modifier != "none":
                    style_terms = {
                        "cinematic": "cinematic, film-quality",
                        "realistic": "realistic, natural, authentic",
                        "ambient": "ambient, atmospheric, immersive",
                        "dramatic": "dramatic, intense, emotional",
                        "peaceful": "peaceful, calm, serene",
                        "energetic": "energetic, dynamic, lively"
                    }
                    if style_modifier in style_terms:
                        text_prompt = f"{style_terms[style_modifier]} {text_prompt}"

                # Enhance prompt if requested
                if enhance_prompt:
                    if conditioning_mode == "enhanced":
                        text_prompt = enhance_audio_prompt(text_prompt)
                    elif conditioning_mode == "standard":
                        pass  # No enhancement
                    else:  # multi_aspect or other
                        text_prompt = enhance_audio_prompt(text_prompt)
                        text_prompt = expand_audio_keywords(text_prompt)
                        text_prompt = emphasize_key_terms(text_prompt)

            print(f"AudioX Enhanced Text-to-Audio: Final prompt: '{text_prompt}'")

            # Validate inputs
            validate_inputs(text_prompt=text_prompt)

            # Get model components
            audiox_model = model["model"]
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # Apply adaptive CFG if enabled
            final_cfg_scale = cfg_scale
            if adaptive_cfg:
                final_cfg_scale = calculate_adaptive_cfg(cfg_scale, text_prompt, "audio")

            # Create conditioning based on mode
            if conditioning_mode == "multi_aspect":
                conditioning = create_multi_aspect_conditioning(text_prompt, int(duration_seconds), device)
            else:
                conditioning = [create_minimal_conditioning(text_prompt, int(duration_seconds), device)]

            # Log negative prompt for future implementation
            if negative_prompt.strip():
                print(f"AudioX Enhanced: Negative prompt noted (not yet implemented): '{negative_prompt}'")

            # Get generation parameters with adaptive CFG
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=final_cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate audio with enhanced conditioning
            with torch.no_grad():
                try:
                    # For now, skip negative conditioning to avoid tensor size mismatches
                    # TODO: Implement proper negative conditioning with matching tensor sizes
                    output = generate_diffusion_cond(
                        copied_model,
                        conditioning=conditioning,
                        device=device,
                        **gen_params
                    )
                except (ValueError, RuntimeError) as e:
                    if "not found in batch metadata" in str(e) or "size of tensor" in str(e):
                        print(f"AudioX Enhanced: Minimal conditioning failed ({e}), trying full conditioning...")
                        # Fall back to full conditioning with empty values
                        empty_values = create_empty_conditioning_values(device, int(duration_seconds))
                        conditioning = [{
                            "text_prompt": text_prompt,
                            "video_prompt": empty_values["video_prompt"],
                            "audio_prompt": empty_values["audio_prompt"],
                            "seconds_start": 0,
                            "seconds_total": int(duration_seconds)
                        }]

                        # Skip negative conditioning in fallback to avoid tensor size issues
                        output = generate_diffusion_cond(
                            copied_model,
                            conditioning=conditioning,
                            device=device,
                            **gen_params
                        )
                    else:
                        raise

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Enhanced audio generation failed: {str(e)}")

class AudioXVideoToMusic:
    """Node for video-to-music generation using AudioX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "video": ("IMAGE",),  # ComfyUI video format
                "text_prompt": ("STRING", {
                    "default": "Generate music for the video",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_music_from_video"
    CATEGORY = "AudioX/Generation"

    def generate_music_from_video(self, model: Dict, video: torch.Tensor, text_prompt: str,
                                 steps: int, cfg_scale: float, seed: int, duration_seconds: float):
        """Generate music from video input."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Validate inputs
            validate_inputs(text_prompt=text_prompt, video_tensor=video)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # CRITICAL FIX: Use specialized video-only conditioning to avoid sequence length conflicts
            print(f"AudioX: Creating video-only conditioning for music generation ({duration_seconds}s)")
            conditioning = create_video_only_conditioning(video, text_prompt, duration_seconds, device)

            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate music
            with torch.no_grad():
                output = generate_diffusion_cond(
                    copied_model,
                    conditioning=[conditioning],
                    device=device,
                    **gen_params
                )

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Video-to-music generation failed: {str(e)}")

class AudioXEnhancedTextToMusic:
    """Enhanced node for text-to-music generation with advanced prompt controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "A peaceful piano melody",
                    "multiline": True,
                    "tooltip": "Describe the music you want to generate"
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale for prompt adherence"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "discordant, harsh, atonal, noise, distorted",
                    "multiline": True,
                    "tooltip": "Negative text prompt (currently logged only - implementation pending)"
                }),
                "music_style": (["none", "classical", "jazz", "electronic", "ambient", "rock", "folk", "cinematic"], {
                    "default": "none",
                    "tooltip": "Musical style to apply"
                }),
                "tempo": (["none", "slow", "moderate", "fast", "very_fast"], {
                    "default": "none",
                    "tooltip": "Tempo indication"
                }),
                "mood": (["none", "happy", "sad", "peaceful", "energetic", "mysterious", "dramatic"], {
                    "default": "none",
                    "tooltip": "Musical mood"
                }),
                "enhance_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically enhance prompt with music-specific keywords"
                }),
                "conditioning_mode": (["standard", "enhanced", "super_enhanced", "multi_aspect"], {
                    "default": "enhanced",
                    "tooltip": "Conditioning enhancement level"
                }),
                "adaptive_cfg": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically adjust CFG based on prompt specificity"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_enhanced_music"
    CATEGORY = "AudioX/Generation"

    def generate_enhanced_music(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float,
                              seed: int, duration_seconds: float, negative_prompt: str = "",
                              music_style: str = "none", tempo: str = "none", mood: str = "none",
                              enhance_prompt: bool = True, conditioning_mode: str = "enhanced",
                              adaptive_cfg: bool = True):
        """Generate music from text prompt with enhanced controls."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Build enhanced prompt with musical attributes
            enhanced_prompt = text_prompt

            # Add musical style
            if music_style != "none":
                style_terms = {
                    "classical": "classical music, orchestral",
                    "jazz": "jazz music, improvised",
                    "electronic": "electronic music, synthesized",
                    "ambient": "ambient music, atmospheric",
                    "rock": "rock music, guitar-driven",
                    "folk": "folk music, acoustic",
                    "cinematic": "cinematic music, film score"
                }
                if music_style in style_terms:
                    enhanced_prompt = f"{style_terms[music_style]} {enhanced_prompt}"

            # Add tempo
            if tempo != "none":
                tempo_terms = {
                    "slow": "slow tempo, relaxed pace",
                    "moderate": "moderate tempo, steady pace",
                    "fast": "fast tempo, energetic pace",
                    "very_fast": "very fast tempo, rapid pace"
                }
                if tempo in tempo_terms:
                    enhanced_prompt = f"{enhanced_prompt}, {tempo_terms[tempo]}"

            # Add mood
            if mood != "none":
                mood_terms = {
                    "happy": "joyful, uplifting, cheerful",
                    "sad": "melancholic, sorrowful, emotional",
                    "peaceful": "calm, serene, tranquil",
                    "energetic": "dynamic, powerful, intense",
                    "mysterious": "enigmatic, suspenseful, dark",
                    "dramatic": "epic, intense, emotional"
                }
                if mood in mood_terms:
                    enhanced_prompt = f"{enhanced_prompt}, {mood_terms[mood]}"

            # Apply advanced conditioning based on mode
            if conditioning_mode == "super_enhanced":
                enhanced_prompt = create_super_enhanced_prompt(enhanced_prompt, "music", "none", True)
            else:
                # Enhance prompt if requested
                if enhance_prompt:
                    if conditioning_mode == "enhanced":
                        enhanced_prompt = enhance_audio_prompt(enhanced_prompt)
                    elif conditioning_mode == "standard":
                        pass  # No enhancement
                    else:  # multi_aspect or other
                        enhanced_prompt = enhance_audio_prompt(enhanced_prompt)
                        enhanced_prompt = expand_audio_keywords(enhanced_prompt)
                        enhanced_prompt = emphasize_key_terms(enhanced_prompt)

                # Ensure it's clearly music
                if "music" not in enhanced_prompt.lower():
                    enhanced_prompt = f"musical {enhanced_prompt}"

            print(f"AudioX Enhanced Text-to-Music: Final prompt: '{enhanced_prompt}'")

            # Validate inputs
            validate_inputs(text_prompt=enhanced_prompt)

            # Get model components
            audiox_model = model["model"]
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # Apply adaptive CFG if enabled
            final_cfg_scale = cfg_scale
            if adaptive_cfg:
                final_cfg_scale = calculate_adaptive_cfg(cfg_scale, enhanced_prompt, "music")

            # Create conditioning based on mode
            if conditioning_mode == "multi_aspect":
                conditioning = create_multi_aspect_conditioning(enhanced_prompt, int(duration_seconds), device)
            else:
                conditioning = [create_minimal_conditioning(enhanced_prompt, int(duration_seconds), device)]

            # Log negative prompt for future implementation
            if negative_prompt.strip():
                print(f"AudioX Enhanced Music: Negative prompt noted (not yet implemented): '{negative_prompt}'")

            # Get generation parameters with adaptive CFG
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=final_cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate music with enhanced conditioning
            with torch.no_grad():
                try:
                    # For now, skip negative conditioning to avoid tensor size mismatches
                    # TODO: Implement proper negative conditioning with matching tensor sizes
                    output = generate_diffusion_cond(
                        copied_model,
                        conditioning=conditioning,
                        device=device,
                        **gen_params
                    )
                except (ValueError, RuntimeError) as e:
                    if "not found in batch metadata" in str(e) or "size of tensor" in str(e):
                        print(f"AudioX Enhanced Music: Minimal conditioning failed ({e}), trying full conditioning...")
                        # Fall back to full conditioning with empty values
                        empty_values = create_empty_conditioning_values(device, int(duration_seconds))
                        conditioning = [{
                            "text_prompt": enhanced_prompt,
                            "video_prompt": empty_values["video_prompt"],
                            "audio_prompt": empty_values["audio_prompt"],
                            "seconds_start": 0,
                            "seconds_total": int(duration_seconds)
                        }]

                        # Skip negative conditioning in fallback to avoid tensor size issues
                        output = generate_diffusion_cond(
                            copied_model,
                            conditioning=conditioning,
                            device=device,
                            **gen_params
                        )
                    else:
                        raise

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Enhanced music generation failed: {str(e)}")

class AudioXMultiModalGeneration:
    """Node for multi-modal audio generation using AudioX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "Generate audio",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "video": ("IMAGE",),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_multimodal_audio"
    CATEGORY = "AudioX/Generation"

    def generate_multimodal_audio(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float,
                                 seed: int, duration_seconds: float, video=None, image=None, audio=None):
        """Generate audio using multiple modalities."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Validate inputs
            validate_inputs(text_prompt=text_prompt, video_tensor=video,
                          image_tensor=image, audio_tensor=audio)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # Prepare conditioning based on available inputs
            conditioning = {
                "text_prompt": text_prompt,
                "seconds_start": 0,
                "seconds_total": int(duration_seconds)
            }

            # Always add empty values for video and audio conditioning to avoid missing key errors
            empty_values = create_empty_conditioning_values(device, int(duration_seconds))
            conditioning["video_prompt"] = empty_values["video_prompt"]  # Will be overridden if video/image provided
            conditioning["audio_prompt"] = empty_values["audio_prompt"]  # Will be overridden if audio provided

            # Handle video input
            if video is not None:
                video_conditioning = prepare_video_conditioning(video, text_prompt, model_config)
                conditioning["video_prompt"] = video_conditioning["video_prompt"]

            # Handle image input (convert to video)
            elif image is not None:
                # Convert image to video
                target_fps = model_config.get("video_fps", 8)
                duration_frames = int(duration_seconds * target_fps)
                video_tensor = convert_image_to_video(image, duration_frames)
                conditioning["video_prompt"] = [video_tensor.unsqueeze(0)]

            # Handle audio input
            if audio is not None:
                conditioning["audio_prompt"] = audio.unsqueeze(0) if audio.dim() == 2 else audio

            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate audio
            with torch.no_grad():
                output = generate_diffusion_cond(
                    copied_model,
                    conditioning=[conditioning],
                    device=device,
                    **gen_params
                )

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Multi-modal audio generation failed: {str(e)}")

class AudioXAudioProcessor:
    """Node for audio processing and utilities."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": True}),
                "target_channels": (["mono", "stereo", "keep"], {"default": "keep"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "sample_rate", "info")
    FUNCTION = "process_audio"
    CATEGORY = "AudioX/Utils"

    def process_audio(self, audio: dict, normalize: bool = True, target_channels: str = "keep"):
        """Process audio tensor."""
        try:
            # Extract waveform and sample rate from AUDIO format
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            processed_audio = waveform.clone()

            # Handle channel conversion
            if target_channels == "mono" and processed_audio.shape[0] > 1:
                processed_audio = torch.mean(processed_audio, dim=0, keepdim=True)
            elif target_channels == "stereo" and processed_audio.shape[0] == 1:
                processed_audio = processed_audio.repeat(2, 1)

            # Normalize if requested
            if normalize:
                max_val = torch.max(torch.abs(processed_audio))
                if max_val > 0:
                    processed_audio = processed_audio / max_val

            # Clamp to valid range
            processed_audio = torch.clamp(processed_audio, -1.0, 1.0)

            # Generate info string
            channels = processed_audio.shape[0]
            duration = processed_audio.shape[1] / sample_rate
            info = f"Channels: {channels}, Duration: {duration:.2f}s, Sample Rate: {sample_rate}Hz"

            # Return in ComfyUI AUDIO format with separate sample rate output
            audio_output = {"waveform": processed_audio, "sample_rate": sample_rate}
            return (audio_output, sample_rate, info)

        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}")

class AudioXVideoMuter:
    """Node for removing audio from video and passing through muted video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),  # ComfyUI video format (batch of images)
            },
            "optional": {
                "preserve_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("muted_video",)
    FUNCTION = "mute_video"
    CATEGORY = "AudioX/Utils"

    def mute_video(self, video: torch.Tensor, preserve_metadata: bool = True):
        """
        Remove audio from video by passing through the video frames without any audio track.
        This is useful for workflows where you want to replace original audio with generated audio.

        Args:
            video: Video tensor in ComfyUI format (batch of images)
            preserve_metadata: Whether to preserve video metadata (frame rate, etc.)

        Returns:
            Tuple containing the muted video (same as input video frames)
        """
        try:
            # Validate input
            if not isinstance(video, torch.Tensor):
                raise ValueError("Video input must be a torch.Tensor")

            if video.dim() != 4:
                raise ValueError(f"Video tensor must be 4-dimensional (batch, height, width, channels), got {video.dim()}D")

            # Simply pass through the video frames - the "muting" happens at the output level
            # where no audio track is associated with these frames
            muted_video = video.clone()

            # Log the operation
            batch_size, height, width, channels = video.shape
            print(f"AudioX Video Muter: Processed {batch_size} frames ({height}x{width}x{channels})")
            print(f"AudioX Video Muter: Video muted - audio track removed")

            return (muted_video,)

        except Exception as e:
            raise RuntimeError(f"Video muting failed: {str(e)}")

class AudioXVideoAudioCombiner:
    """Node for combining muted video with generated audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),  # ComfyUI video format (batch of images)
                "audio": ("AUDIO",),  # ComfyUI audio format
            },
            "optional": {
                "sync_duration": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically sync audio duration to video duration"
                }),
                "loop_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loop audio if shorter than video"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("video", "synced_audio", "info")
    FUNCTION = "combine_video_audio"
    CATEGORY = "AudioX/Utils"

    def combine_video_audio(self, video: torch.Tensor, audio: dict,
                           sync_duration: bool = True, loop_audio: bool = False):
        """
        Combine video frames with audio, ensuring proper synchronization.

        Args:
            video: Video tensor in ComfyUI format
            audio: Audio dictionary with 'waveform' and 'sample_rate'
            sync_duration: Whether to sync audio duration to video
            loop_audio: Whether to loop audio if it's shorter than video

        Returns:
            Tuple containing (video, synced_audio, info_string)
        """
        try:
            # Validate inputs
            if not isinstance(video, torch.Tensor):
                raise ValueError("Video input must be a torch.Tensor")

            if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
                raise ValueError("Audio input must be a dictionary with 'waveform' and 'sample_rate' keys")

            # Get video and audio properties
            num_frames = video.shape[0]
            audio_waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # Assume standard video frame rate (can be made configurable)
            video_fps = 8.0  # Default FPS for ComfyUI video workflows
            video_duration = num_frames / video_fps
            audio_duration = audio_waveform.shape[-1] / sample_rate

            print(f"AudioX Combiner: Video frames: {num_frames}, duration: {video_duration:.2f}s")
            print(f"AudioX Combiner: Audio duration: {audio_duration:.2f}s")

            synced_audio = audio.copy()

            if sync_duration:
                target_samples = int(video_duration * sample_rate)
                current_samples = audio_waveform.shape[-1]

                if current_samples < target_samples:
                    if loop_audio:
                        # Loop audio to match video duration
                        repeat_count = (target_samples + current_samples - 1) // current_samples
                        looped_waveform = audio_waveform.repeat(1, 1, repeat_count)
                        synced_audio['waveform'] = looped_waveform[:, :, :target_samples]
                        print(f"AudioX Combiner: Audio looped {repeat_count} times to match video duration")
                    else:
                        # Pad audio with silence
                        padding_samples = target_samples - current_samples
                        padding = torch.zeros(audio_waveform.shape[0], audio_waveform.shape[1],
                                            padding_samples, device=audio_waveform.device,
                                            dtype=audio_waveform.dtype)
                        synced_audio['waveform'] = torch.cat([audio_waveform, padding], dim=-1)
                        print(f"AudioX Combiner: Audio padded with {padding_samples} samples of silence")

                elif current_samples > target_samples:
                    # Trim audio to match video duration
                    synced_audio['waveform'] = audio_waveform[:, :, :target_samples]
                    print(f"AudioX Combiner: Audio trimmed to match video duration")

            # Generate info string
            final_audio_duration = synced_audio['waveform'].shape[-1] / sample_rate
            info = (f"Video: {num_frames} frames ({video_duration:.2f}s), "
                   f"Audio: {final_audio_duration:.2f}s, "
                   f"Sample Rate: {sample_rate}Hz")

            return (video, synced_audio, info)

        except Exception as e:
            raise RuntimeError(f"Video-audio combination failed: {str(e)}")

class AudioXVolumeControl:
    """Node for controlling audio volume/gain with a slider for proper gain staging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "volume_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -60.0,
                    "max": 20.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Volume adjustment in decibels (dB). 0dB = no change, +6dB = double volume, -6dB = half volume"
                }),
            },
            "optional": {
                "step_size": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Step size for volume slider precision (0.01 = very fine, 1.0 = coarse)"
                }),
                "normalize_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Normalize audio after volume adjustment to prevent clipping"
                }),
                "soft_limit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply soft limiting to prevent harsh clipping"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "peak_level_db", "info")
    FUNCTION = "adjust_volume"
    CATEGORY = "AudioX/Utils"

    def adjust_volume(self, audio: dict, volume_db: float = 0.0, step_size: float = 0.1,
                     normalize_after: bool = False, soft_limit: bool = True):
        """
        Adjust audio volume with proper gain staging.

        Args:
            audio: Audio dictionary with 'waveform' and 'sample_rate'
            volume_db: Volume adjustment in decibels
            step_size: Step size for volume precision (affects rounding)
            normalize_after: Whether to normalize after volume adjustment
            soft_limit: Whether to apply soft limiting

        Returns:
            Tuple containing (adjusted_audio, peak_level_db, info_string)
        """
        try:
            # Extract waveform and sample rate from AUDIO format
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            # Round volume_db to the specified step size for precision
            rounded_volume_db = round(volume_db / step_size) * step_size

            # Log the volume adjustment
            if abs(volume_db - rounded_volume_db) > 0.001:
                print(f"AudioX Volume Control: Rounded {volume_db:.3f}dB to {rounded_volume_db:.3f}dB (step: {step_size:.3f}dB)")
            else:
                print(f"AudioX Volume Control: Applying {rounded_volume_db:+.3f}dB volume adjustment")

            # Convert dB to linear gain
            # Formula: linear_gain = 10^(dB/20)
            linear_gain = 10.0 ** (rounded_volume_db / 20.0)

            # Apply gain to the waveform
            adjusted_waveform = waveform * linear_gain

            # Calculate peak level before any limiting
            peak_linear = torch.max(torch.abs(adjusted_waveform)).item()
            peak_db = 20 * torch.log10(torch.tensor(peak_linear + 1e-10)).item()

            # Apply soft limiting if enabled and signal is clipping
            if soft_limit and peak_linear > 1.0:
                # Soft limiting using tanh function
                adjusted_waveform = torch.tanh(adjusted_waveform * 0.9)
                print(f"AudioX Volume Control: Soft limiting applied (peak was {peak_db:.1f}dB)")

            # Normalize after adjustment if requested
            elif normalize_after:
                max_val = torch.max(torch.abs(adjusted_waveform))
                if max_val > 0:
                    adjusted_waveform = adjusted_waveform / max_val
                    print(f"AudioX Volume Control: Normalized after volume adjustment")

            # Hard clamp as final safety measure
            adjusted_waveform = torch.clamp(adjusted_waveform, -1.0, 1.0)

            # Calculate final peak level
            final_peak_linear = torch.max(torch.abs(adjusted_waveform)).item()
            final_peak_db = 20 * torch.log10(torch.tensor(final_peak_linear + 1e-10)).item()

            # Generate info string
            channels = adjusted_waveform.shape[1] if adjusted_waveform.dim() == 3 else adjusted_waveform.shape[0]
            duration = adjusted_waveform.shape[-1] / sample_rate

            # Show both original and rounded volume if they differ
            volume_display = f"{rounded_volume_db:+.2f}dB"
            if abs(volume_db - rounded_volume_db) > 0.001:
                volume_display = f"{volume_db:+.2f}→{rounded_volume_db:+.2f}dB"

            info = (f"Volume: {volume_display} (x{linear_gain:.3f}), "
                   f"Peak: {final_peak_db:.1f}dB, "
                   f"Step: {step_size:.2f}dB, "
                   f"Duration: {duration:.2f}s, "
                   f"Channels: {channels}")

            # Return in ComfyUI AUDIO format
            audio_output = {"waveform": adjusted_waveform, "sample_rate": sample_rate}
            return (audio_output, final_peak_db, info)

        except Exception as e:
            raise RuntimeError(f"Volume control failed: {str(e)}")

class AudioXAdvancedVolumeControl:
    """Advanced volume control node with additional professional audio features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "volume_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -80.0,
                    "max": 30.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Volume adjustment in decibels (dB)"
                }),
            },
            "optional": {
                "step_size": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 5.0,
                    "step": 0.001,
                    "tooltip": "Step size for volume slider precision"
                }),
                "target_lufs": ("FLOAT", {
                    "default": -23.0,
                    "min": -40.0,
                    "max": -10.0,
                    "step": 0.1,
                    "tooltip": "Target LUFS for automatic loudness normalization (-23 LUFS is broadcast standard)"
                }),
                "auto_normalize_lufs": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically normalize to target LUFS instead of using volume_db"
                }),
                "limiter_threshold_db": ("FLOAT", {
                    "default": -1.0,
                    "min": -10.0,
                    "max": 0.0,
                    "step": 0.1,
                    "tooltip": "Limiter threshold in dB to prevent clipping"
                }),
                "limiter_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable soft limiting to prevent clipping"
                }),
                "fade_in_ms": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5000.0,
                    "step": 10.0,
                    "tooltip": "Fade in duration in milliseconds"
                }),
                "fade_out_ms": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5000.0,
                    "step": 10.0,
                    "tooltip": "Fade out duration in milliseconds"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "peak_level_db", "lufs_level", "info")
    FUNCTION = "advanced_volume_control"
    CATEGORY = "AudioX/Utils"

    def advanced_volume_control(self, audio: dict, volume_db: float = 0.0, step_size: float = 0.1,
                               target_lufs: float = -23.0, auto_normalize_lufs: bool = False,
                               limiter_threshold_db: float = -1.0, limiter_enabled: bool = True,
                               fade_in_ms: float = 0.0, fade_out_ms: float = 0.0):
        """
        Advanced volume control with professional audio features.
        """
        try:
            # Extract waveform and sample rate
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            # Round volume_db to step size
            rounded_volume_db = round(volume_db / step_size) * step_size

            # Calculate LUFS (simplified approximation)
            rms = torch.sqrt(torch.mean(waveform ** 2))
            current_lufs = 20 * torch.log10(rms + 1e-10) - 0.691  # Rough LUFS approximation

            if auto_normalize_lufs:
                # Calculate gain needed to reach target LUFS
                lufs_gain_db = target_lufs - current_lufs.item()
                final_volume_db = lufs_gain_db
                print(f"AudioX Advanced Volume: Auto-normalizing from {current_lufs:.1f} to {target_lufs:.1f} LUFS ({lufs_gain_db:+.1f}dB)")
            else:
                final_volume_db = rounded_volume_db
                print(f"AudioX Advanced Volume: Manual volume adjustment {final_volume_db:+.3f}dB")

            # Apply volume adjustment
            linear_gain = 10.0 ** (final_volume_db / 20.0)
            adjusted_waveform = waveform * linear_gain

            # Apply fades if specified
            if fade_in_ms > 0 or fade_out_ms > 0:
                adjusted_waveform = self._apply_fades(adjusted_waveform, sample_rate, fade_in_ms, fade_out_ms)

            # Apply limiting if enabled
            peak_before_limiting = torch.max(torch.abs(adjusted_waveform)).item()
            if limiter_enabled:
                limiter_threshold_linear = 10.0 ** (limiter_threshold_db / 20.0)
                adjusted_waveform = self._apply_soft_limiter(adjusted_waveform, limiter_threshold_linear)

            # Calculate final levels
            final_peak = torch.max(torch.abs(adjusted_waveform)).item()
            final_peak_db = 20 * torch.log10(final_peak + 1e-10)

            final_rms = torch.sqrt(torch.mean(adjusted_waveform ** 2))
            final_lufs = 20 * torch.log10(final_rms + 1e-10) - 0.691

            # Generate comprehensive info
            info_parts = []
            if auto_normalize_lufs:
                info_parts.append(f"LUFS: {current_lufs:.1f}→{final_lufs:.1f}")
            else:
                info_parts.append(f"Volume: {final_volume_db:+.2f}dB")

            info_parts.extend([
                f"Peak: {final_peak_db:.1f}dB",
                f"LUFS: {final_lufs:.1f}",
                f"Step: {step_size:.3f}dB"
            ])

            if fade_in_ms > 0 or fade_out_ms > 0:
                info_parts.append(f"Fades: {fade_in_ms:.0f}ms/{fade_out_ms:.0f}ms")

            if limiter_enabled and peak_before_limiting > 10.0 ** (limiter_threshold_db / 20.0):
                info_parts.append("Limited")

            info = ", ".join(info_parts)

            # Return results
            audio_output = {"waveform": adjusted_waveform, "sample_rate": sample_rate}
            return (audio_output, final_peak_db, final_lufs.item(), info)

        except Exception as e:
            raise RuntimeError(f"Advanced volume control failed: {str(e)}")

    def _apply_fades(self, waveform: torch.Tensor, sample_rate: int, fade_in_ms: float, fade_out_ms: float):
        """Apply fade in/out to waveform."""
        fade_in_samples = int(fade_in_ms * sample_rate / 1000.0)
        fade_out_samples = int(fade_out_ms * sample_rate / 1000.0)

        total_samples = waveform.shape[-1]
        faded_waveform = waveform.clone()

        # Apply fade in
        if fade_in_samples > 0 and fade_in_samples < total_samples:
            fade_in_curve = torch.linspace(0, 1, fade_in_samples)
            faded_waveform[..., :fade_in_samples] *= fade_in_curve

        # Apply fade out
        if fade_out_samples > 0 and fade_out_samples < total_samples:
            fade_out_curve = torch.linspace(1, 0, fade_out_samples)
            faded_waveform[..., -fade_out_samples:] *= fade_out_curve

        return faded_waveform

    def _apply_soft_limiter(self, waveform: torch.Tensor, threshold: float):
        """Apply soft limiting using tanh compression."""
        # Only apply limiting to samples above threshold
        mask = torch.abs(waveform) > threshold
        limited_waveform = waveform.clone()

        # Apply tanh limiting to samples above threshold
        over_threshold = waveform[mask]
        sign = torch.sign(over_threshold)
        magnitude = torch.abs(over_threshold)

        # Soft limiting curve: threshold + (1-threshold) * tanh((magnitude-threshold)/(1-threshold))
        limited_magnitude = threshold + (1 - threshold) * torch.tanh((magnitude - threshold) / (1 - threshold))
        limited_waveform[mask] = sign * limited_magnitude

        return limited_waveform

class AudioXPromptHelper:
    """Node for helping users create better audio prompts with templates and enhancement."""

    @classmethod
    def INPUT_TYPES(cls):
        templates = get_audio_prompt_templates()
        template_options = ["none"]
        for category, subcategories in templates.items():
            for subcategory in subcategories.keys():
                template_options.append(f"{category}_{subcategory}")

        return {
            "required": {
                "base_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Your base audio description"
                }),
                "template": (template_options, {
                    "default": "none",
                    "tooltip": "Choose a template to enhance your prompt"
                }),
                "enhance_automatically": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically add audio-specific keywords"
                }),
                "add_quality_terms": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add quality enhancement terms"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "muffled, distorted, low quality, noise",
                    "multiline": True,
                    "tooltip": "What to avoid in the audio"
                }),
                "style_modifier": (["none", "cinematic", "realistic", "ambient", "dramatic", "peaceful", "energetic"], {
                    "default": "none",
                    "tooltip": "Add style modifier to the prompt"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt", "prompt_info")
    FUNCTION = "create_enhanced_prompt"
    CATEGORY = "AudioX/Utils"

    def create_enhanced_prompt(self, base_prompt: str, template: str, enhance_automatically: bool,
                             add_quality_terms: bool, negative_prompt: str = "", style_modifier: str = "none"):
        """Create an enhanced audio prompt with templates and improvements."""
        try:
            enhanced_prompt = base_prompt.strip()

            # Apply template if selected
            if template != "none":
                templates = get_audio_prompt_templates()
                template_parts = template.split('_')
                if len(template_parts) == 2:
                    category, subcategory = template_parts
                    if category in templates and subcategory in templates[category]:
                        template_text = templates[category][subcategory]
                        if enhanced_prompt:
                            enhanced_prompt = f"{enhanced_prompt}, {template_text}"
                        else:
                            enhanced_prompt = template_text

            # Add style modifier
            if style_modifier != "none":
                style_terms = {
                    "cinematic": "cinematic, film-quality",
                    "realistic": "realistic, natural, authentic",
                    "ambient": "ambient, atmospheric, immersive",
                    "dramatic": "dramatic, intense, emotional",
                    "peaceful": "peaceful, calm, serene",
                    "energetic": "energetic, dynamic, lively"
                }
                if style_modifier in style_terms:
                    enhanced_prompt = f"{style_terms[style_modifier]} {enhanced_prompt}"

            # Enhance automatically if requested
            if enhance_automatically:
                enhanced_prompt = enhance_audio_prompt(enhanced_prompt)

            # Add quality terms
            if add_quality_terms:
                quality_terms = "high quality, clear, well-defined"
                enhanced_prompt = f"{enhanced_prompt}, {quality_terms}"

            # Clean up the prompt
            enhanced_prompt = enhanced_prompt.strip().strip(',').strip()

            # Ensure negative prompt has good defaults
            if not negative_prompt.strip():
                negative_prompt = "muffled, distorted, low quality, noise, static"

            # Create info string
            modifications = []
            if template != "none":
                modifications.append(f"Template: {template}")
            if style_modifier != "none":
                modifications.append(f"Style: {style_modifier}")
            if enhance_automatically:
                modifications.append("Auto-enhanced")
            if add_quality_terms:
                modifications.append("Quality terms added")

            prompt_info = f"Modifications: {', '.join(modifications) if modifications else 'None'}"

            return (enhanced_prompt, negative_prompt, prompt_info)

        except Exception as e:
            raise RuntimeError(f"Prompt enhancement failed: {str(e)}")




