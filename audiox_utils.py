import os
import torch
import torchaudio
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
import logging




# Global model cache
_model_cache = {}

def get_device():
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test if MPS actually works by creating a small tensor
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            return "mps"
        except Exception as e:
            print(f"AudioX: MPS detected but not functional: {e}")
            print("AudioX: Falling back to CPU")
            return "cpu"
    else:
        return "cpu"

def load_audiox_model_from_file(model_path: str, config_path: Optional[str] = None, device: Optional[str] = None):
    """
    Load AudioX model from local files.

    Args:
        model_path: Path to model file (.ckpt, .safetensors, .pt)
        config_path: Path to config file (.json), optional
        device: Device to load model on

    Returns:
        Tuple of (model, model_config)
    """
    if device is None:
        device = get_device()

    cache_key = f"{model_path}_{device}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import json

        # Ensure dependencies are available before importing AudioX components
        try:
            import einops_exts
            import dac
            print("AudioX: All dependencies available")
        except ImportError as dep_error:
            print(f"AudioX: Dependency error: {dep_error}")
            # Try to install missing dependencies
            try:
                import subprocess
                import sys
                if "einops_exts" in str(dep_error):
                    print("AudioX: Installing einops-exts...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops-exts"])
                elif "dac" in str(dep_error):
                    print("AudioX: Installing descript-audio-codec...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "descript-audio-codec"])

                # Try importing again
                import einops_exts
                import dac
                print("AudioX: Dependencies installed and available")
            except Exception as install_error:
                print(f"AudioX: Failed to install dependencies: {install_error}")
                raise RuntimeError(f"Missing required dependencies: {dep_error}")

        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        print(f"Loading AudioX model from: {model_path}")

        # Load config
        if config_path and os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"Warning: Config file {config_path} is empty, using default config")
                        model_config = get_default_audiox_config()
                    else:
                        model_config = json.loads(content)
                        print(f"Config loaded successfully: {len(content)} characters")
                        # Validate essential config fields
                        if not model_config.get('model_type'):
                            print("Warning: Config missing model_type, using default")
                            model_config = get_default_audiox_config()
                        elif not model_config.get('model', {}).get('diffusion'):
                            print("Warning: Config missing diffusion section, using default")
                            model_config = get_default_audiox_config()
            except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
                print(f"Warning: Failed to parse config file {config_path}: {e}")
                print("Using default AudioX config")
                model_config = get_default_audiox_config()
        else:
            if config_path:
                print(f"Warning: Config file not found: {config_path}")
            print("Using default AudioX config")
            model_config = get_default_audiox_config()

        # EMERGENCY: Check and install vector_quantize_pytorch if missing
        print("EMERGENCY: Checking vector_quantize_pytorch...")
        try:
            from vector_quantize_pytorch import ResidualVQ, FSQ
            print("✓ vector_quantize_pytorch available")
        except ImportError as e:
            print(f"✗ vector_quantize_pytorch missing: {e}")
            print("EMERGENCY: Installing vector_quantize_pytorch NOW...")
            import subprocess
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "vector-quantize-pytorch"
                ], capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    print("✅ vector_quantize_pytorch installed successfully!")
                    # Try import again
                    from vector_quantize_pytorch import ResidualVQ, FSQ
                    print("✓ vector_quantize_pytorch now working")
                else:
                    print(f"❌ Failed to install: {result.stderr}")
                    raise RuntimeError(f"Failed to install vector_quantize_pytorch: {result.stderr}")
            except Exception as install_error:
                print(f"❌ Installation error: {install_error}")
                raise RuntimeError(f"Failed to install vector_quantize_pytorch: {install_error}")

        # Check other dependencies before creating model
        print("Checking other dependencies...")
        try:
            import sys
            print(f"Python executable: {sys.executable}")
            print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

            # Test critical imports
            from x_transformers import ContinuousTransformerWrapper, Encoder
            print("✓ x_transformers import successful")

            import einops_exts
            print("✓ einops_exts import successful")

            import dac
            print("✓ dac import successful")

            import alias_free_torch
            print("✓ alias_free_torch import successful")

        except ImportError as e:
            print(f"✗ Dependency check failed: {e}")
            print("Available packages:")
            import pkg_resources
            installed_packages = [d.project_name for d in pkg_resources.working_set]
            for pkg in ['x-transformers', 'einops-exts', 'descript-audio-codec', 'alias-free-torch']:
                status = "✓" if pkg in installed_packages else "✗"
                print(f"  {status} {pkg}")
            raise RuntimeError(f"Missing dependency: {e}")

        # Create model from config
        print("Creating model from config...")
        model = create_model_from_config(model_config)

        # Load weights
        print(f"Loading model weights from {model_path}...")
        try:
            state_dict = load_ckpt_state_dict(model_path)
            print(f"Loaded state dict with {len(state_dict)} keys")

            # Load state dict into model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in model: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # Show first 5

        except Exception as e:
            print(f"Error loading state dict: {e}")
            # Try alternative loading methods
            try:
                import torch
                print("Trying direct torch.load...")
                checkpoint = torch.load(model_path, map_location='cpu')

                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded with torch.load")

            except Exception as e2:
                raise RuntimeError(f"Failed to load model weights with both methods: {e}, {e2}")

        # Move to device and set eval mode
        model = model.to(device)
        model.eval()

        _model_cache[cache_key] = (model, model_config)
        print(f"AudioX model loaded successfully on {device}")

        return model, model_config

    except Exception as e:
        raise RuntimeError(f"Failed to load AudioX model from file: {str(e)}")

def get_default_audiox_config():
    """Get default configuration for AudioX model."""
    return {
        "model_type": "diffusion_cond",
        "sample_size": 485100,
        "sample_rate": 44100,
        "video_fps": 5,
        "audio_channels": 2,
        "model": {
            "pretransform": {
                "type": "autoencoder",
                "iterate_batch": True,
                "config": {
                    "encoder": {
                        "type": "oobleck",
                        "requires_grad": False,
                        "config": {
                            "in_channels": 2,
                            "channels": 128,
                            "c_mults": [1, 2, 4, 8, 16],
                            "strides": [2, 4, 4, 8, 8],
                            "latent_dim": 128,
                            "use_snake": True
                        }
                    },
                    "decoder": {
                        "type": "oobleck",
                        "config": {
                            "out_channels": 2,
                            "channels": 128,
                            "c_mults": [1, 2, 4, 8, 16],
                            "strides": [2, 4, 4, 8, 8],
                            "latent_dim": 64,
                            "use_snake": True,
                            "final_tanh": False
                        }
                    },
                    "bottleneck": {
                        "type": "vae"
                    },
                    "latent_dim": 64,
                    "downsampling_ratio": 2048,
                    "io_channels": 2
                }
            },
            "conditioning": {
                "configs": [
                    {
                        "id": "video_prompt",
                        "type": "clip",
                        "config": {
                            "clip_model_name": "clip-vit-base-patch32"
                        }
                    },
                    {
                        "id": "text_prompt",
                        "type": "t5",
                        "config": {
                            "t5_model_name": "t5-base",
                            "max_length": 128
                        }
                    },
                    {
                        "id": "audio_prompt",
                        "type": "audio_autoencoder",
                        "config": {
                            "sample_rate": 44100,
                            "pretransform_config": {
                                "type": "autoencoder",
                                "iterate_batch": True,
                                "config": {
                                    "encoder": {
                                        "type": "oobleck",
                                        "requires_grad": False,
                                        "config": {
                                            "in_channels": 2,
                                            "channels": 128,
                                            "c_mults": [1, 2, 4, 8, 16],
                                            "strides": [2, 4, 4, 8, 8],
                                            "latent_dim": 128,
                                            "use_snake": True
                                        }
                                    },
                                    "decoder": {
                                        "type": "oobleck",
                                        "config": {
                                            "out_channels": 2,
                                            "channels": 128,
                                            "c_mults": [1, 2, 4, 8, 16],
                                            "strides": [2, 4, 4, 8, 8],
                                            "latent_dim": 64,
                                            "use_snake": True,
                                            "final_tanh": False
                                        }
                                    },
                                    "bottleneck": {
                                        "type": "vae"
                                    },
                                    "latent_dim": 64,
                                    "downsampling_ratio": 2048,
                                    "io_channels": 2
                                }
                            }
                        }
                    }
                ],
                "cond_dim": 768
            },
            "diffusion": {
                "cross_attention_cond_ids": ["video_prompt", "text_prompt", "audio_prompt"],
                "global_cond_ids": [],
                "type": "dit",
                "config": {
                    "io_channels": 64,
                    "embed_dim": 1536,
                    "depth": 24,
                    "num_heads": 24,
                    "cond_token_dim": 768,
                    "global_cond_dim": 1536,
                    "project_cond_tokens": False,
                    "transformer_type": "continuous_transformer",
                    "video_fps": 5
                }
            },
            "io_channels": 64
        },

    }

def prepare_text_conditioning(text_prompt: str, model_config: Dict) -> Dict:
    """Prepare text conditioning for AudioX generation."""
    return {
        "text_prompt": text_prompt,
        "seconds_start": 0,
        "seconds_total": 10  # Default 10 seconds
    }

def prepare_video_conditioning(video_tensor: torch.Tensor, text_prompt: str, model_config: Dict) -> Dict:
    """
    Prepare video conditioning for AudioX generation.

    Args:
        video_tensor: Video tensor in ComfyUI format [B, F, H, W, C] or [F, H, W, C]
        text_prompt: Text description
        model_config: Model configuration

    Returns:
        Conditioning dictionary
    """
    print(f"AudioX: Input video tensor shape: {video_tensor.shape}")

    # Handle different video tensor formats from ComfyUI
    if video_tensor.dim() == 5:
        # Format: [B, F, H, W, C] - remove batch dimension
        video_tensor = video_tensor.squeeze(0)  # Now [F, H, W, C]
    elif video_tensor.dim() == 4:
        # Format: [F, H, W, C] - already correct
        pass
    else:
        raise ValueError(f"Unexpected video tensor dimensions: {video_tensor.shape}")

    # Ensure we have the right number of channels (3 for RGB)
    if video_tensor.shape[-1] != 3:
        print(f"AudioX: Warning - video has {video_tensor.shape[-1]} channels, expected 3 (RGB)")
        if video_tensor.shape[-1] > 3:
            # Take only the first 3 channels
            video_tensor = video_tensor[..., :3]
            print(f"AudioX: Truncated to RGB channels, new shape: {video_tensor.shape}")
        else:
            # Repeat channels to get 3
            video_tensor = video_tensor.repeat(1, 1, 1, 3 // video_tensor.shape[-1] + 1)[..., :3]
            print(f"AudioX: Expanded to RGB channels, new shape: {video_tensor.shape}")

    # CRITICAL FIX: Standardize video to expected frame count
    # The model expects exactly 50 frames (10 seconds * 5 fps) to match audio conditioning sequence length
    expected_frames = 50
    current_frames = video_tensor.shape[0]

    print(f"AudioX: Current frames: {current_frames}, Expected frames: {expected_frames}")

    if current_frames != expected_frames:
        if current_frames > expected_frames:
            # Truncate to expected frames
            video_tensor = video_tensor[:expected_frames]
            print(f"AudioX: Truncated video from {current_frames} to {expected_frames} frames")
        else:
            # Pad by repeating the last frame
            last_frame = video_tensor[-1:].repeat(expected_frames - current_frames, 1, 1, 1)
            video_tensor = torch.cat([video_tensor, last_frame], dim=0)
            print(f"AudioX: Padded video from {current_frames} to {expected_frames} frames")

    # Convert from [F, H, W, C] to [F, C, H, W] for CLIP processing
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    print(f"AudioX: Video tensor after permute: {video_tensor.shape}")

    # Ensure video tensor is in the right format for CLIP (values should be 0-255 or 0-1)
    if video_tensor.max() <= 1.0 and video_tensor.min() >= 0.0:
        # Convert from [0,1] to [0,255] range for consistency
        video_tensor = video_tensor * 255.0
        print(f"AudioX: Converted video tensor to 0-255 range")

    # Add batch dimension and create final conditioning
    final_video_tensor = video_tensor.unsqueeze(0)  # [1, F, C, H, W]
    print(f"AudioX: Final video conditioning tensor shape: {final_video_tensor.shape}")

    conditioning = {
        "video_prompt": [final_video_tensor],
        "text_prompt": text_prompt,
        "seconds_start": 0,
        "seconds_total": 10
    }

    print(f"AudioX: Video conditioning created with keys: {list(conditioning.keys())}")
    return conditioning

def prepare_audio_conditioning(audio_tensor: Optional[torch.Tensor], text_prompt: str, model_config: Dict) -> Dict:
    """Prepare audio conditioning for AudioX generation."""
    conditioning = {
        "text_prompt": text_prompt,
        "video_prompt": None,
        "seconds_start": 0,
        "seconds_total": 10
    }
    
    if audio_tensor is not None:
        conditioning["audio_prompt"] = audio_tensor.unsqueeze(0)  # Add batch dimension
    else:
        conditioning["audio_prompt"] = None
        
    return conditioning

def convert_image_to_video(image_tensor: torch.Tensor, duration_frames: int = 30) -> torch.Tensor:
    """
    Convert a single image to a video by repeating frames.
    
    Args:
        image_tensor: Image tensor [B, H, W, C] or [H, W, C]
        duration_frames: Number of frames to create
        
    Returns:
        Video tensor [F, C, H, W]
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove batch dim
    
    # Convert [H, W, C] to [C, H, W]
    if image_tensor.shape[-1] == 3:  # Channels last
        image_tensor = image_tensor.permute(2, 0, 1)
    
    # Repeat the image for the specified duration
    video_tensor = image_tensor.unsqueeze(0).repeat(duration_frames, 1, 1, 1)
    
    return video_tensor

def postprocess_audio_output(audio_tensor: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, Dict]:
    """
    Postprocess AudioX output for ComfyUI compatibility.

    Args:
        audio_tensor: Generated audio tensor
        sample_rate: Sample rate of the audio

    Returns:
        Tuple of (processed_audio, metadata) where processed_audio is [batch, channels, samples]
    """
    # Ensure audio is in the right format
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension if present

    # Ensure we have at least 2 dimensions [channels, samples]
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension

    # Normalize audio
    audio_tensor = audio_tensor.to(torch.float32)
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 0:
        audio_tensor = audio_tensor / max_val

    # Clamp to valid range
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)

    # Add batch dimension for ComfyUI compatibility [batch, channels, samples]
    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.unsqueeze(0)

    metadata = {
        "sample_rate": sample_rate,
        "channels": audio_tensor.shape[1] if audio_tensor.dim() == 3 else 1,
        "duration": audio_tensor.shape[-1] / sample_rate,
        "format": "float32"
    }

    return audio_tensor, metadata

def validate_inputs(text_prompt: Optional[str] = None, 
                   video_tensor: Optional[torch.Tensor] = None,
                   image_tensor: Optional[torch.Tensor] = None,
                   audio_tensor: Optional[torch.Tensor] = None) -> bool:
    """Validate that at least one conditioning input is provided."""
    inputs_provided = [
        text_prompt is not None and len(text_prompt.strip()) > 0,
        video_tensor is not None,
        image_tensor is not None,
        audio_tensor is not None
    ]
    
    if not any(inputs_provided):
        raise ValueError("At least one conditioning input (text, video, image, or audio) must be provided")
    
    return True

def get_generation_parameters(steps: int = 250, 
                            cfg_scale: float = 7.0,
                            seed: int = -1,
                            sample_size: int = 2097152,
                            sigma_min: float = 0.3,
                            sigma_max: float = 500.0,
                            sampler_type: str = "dpmpp-3m-sde") -> Dict:
    """Get standardized generation parameters."""
    return {
        "steps": max(1, min(1000, steps)),
        "cfg_scale": max(0.1, min(20.0, cfg_scale)),
        "seed": seed,
        "sample_size": sample_size,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sampler_type": sampler_type
    }

def create_minimal_conditioning(text_prompt: str, duration_seconds: int, device: str = "cpu") -> Dict:
    """
    Create minimal conditioning that includes all required keys for text-to-audio generation.
    Note: This model requires both video_prompt and audio_prompt even for text-only generation.

    Args:
        text_prompt: The text prompt for generation
        duration_seconds: Duration for the generation
        device: Device to create tensors on

    Returns:
        Dictionary with conditioning including empty video and audio tensors
    """
    # Get empty values for video and audio conditioning
    empty_values = create_empty_conditioning_values(device, duration_seconds)

    return {
        "text_prompt": text_prompt,
        "video_prompt": empty_values["video_prompt"],
        "audio_prompt": empty_values["audio_prompt"],
        "seconds_start": 0,
        "seconds_total": int(duration_seconds)
    }

def create_video_only_conditioning(video_tensor: torch.Tensor, text_prompt: str,
                                 duration_seconds: int, device: str = "cpu") -> Dict:
    """
    Create conditioning specifically for video-to-audio generation.
    This includes empty audio conditioning to satisfy conditioner requirements.

    Args:
        video_tensor: Video tensor for conditioning
        text_prompt: Text prompt for generation
        duration_seconds: Duration for the generation
        device: Device to create tensors on

    Returns:
        Dictionary with video, text, and empty audio conditioning
    """
    print(f"AudioX: Creating video-only conditioning for {duration_seconds}s generation")

    # Prepare video conditioning with proper frame standardization
    video_conditioning = prepare_video_conditioning(video_tensor, text_prompt, {})

    # Get empty audio conditioning to satisfy conditioner requirements
    empty_values = create_empty_conditioning_values(device, duration_seconds)

    # Create final conditioning with video, text, and empty audio
    conditioning = {
        "text_prompt": text_prompt,
        "video_prompt": video_conditioning["video_prompt"],
        "audio_prompt": empty_values["audio_prompt"],  # Required by conditioner
        "seconds_start": 0,
        "seconds_total": int(duration_seconds)
    }

    print(f"AudioX: Video-only conditioning keys: {list(conditioning.keys())}")
    print(f"AudioX: Video prompt shape: {conditioning['video_prompt'][0].shape}")

    return conditioning

def create_enhanced_video_conditioning(video_tensor: torch.Tensor, text_prompt: str,
                                     duration_seconds: int, device: str = "cpu",
                                     text_weight: float = 1.0, video_weight: float = 1.0,
                                     negative_prompt: str = "") -> Dict:
    """
    Create enhanced conditioning for video-to-audio generation with better text-video balance.

    Args:
        video_tensor: Video tensor for conditioning
        text_prompt: Text prompt for generation
        duration_seconds: Duration for the generation
        device: Device to create tensors on
        text_weight: Weight for text conditioning influence
        video_weight: Weight for video conditioning influence
        negative_prompt: Negative text prompt to avoid

    Returns:
        Dictionary with enhanced conditioning
    """
    print(f"AudioX: Creating enhanced video conditioning (text_weight={text_weight}, video_weight={video_weight})")

    # Enhance text prompt with audio-specific processing
    enhanced_text = enhance_audio_prompt(text_prompt)
    print(f"AudioX: Enhanced text prompt: '{enhanced_text}'")

    # Prepare video conditioning
    video_conditioning = prepare_video_conditioning(video_tensor, enhanced_text, {})

    # Get empty audio conditioning
    empty_values = create_empty_conditioning_values(device, duration_seconds)

    # Create conditioning with weights
    conditioning = {
        "text_prompt": enhanced_text,
        "video_prompt": video_conditioning["video_prompt"],
        "audio_prompt": empty_values["audio_prompt"],
        "seconds_start": 0,
        "seconds_total": int(duration_seconds),
        # Add weight metadata for potential use in generation
        "_text_weight": text_weight,
        "_video_weight": video_weight,
        "_negative_prompt": negative_prompt
    }

    return conditioning

def enhance_audio_prompt(text_prompt: str) -> str:
    """
    Enhance text prompt for better audio generation by adding audio-specific context.

    Args:
        text_prompt: Original text prompt

    Returns:
        Enhanced text prompt with better audio context
    """
    if not text_prompt or len(text_prompt.strip()) == 0:
        return "ambient audio, natural sounds"

    # Audio-specific keywords that help with generation
    audio_keywords = {
        'music': ['musical', 'melody', 'rhythm', 'harmony'],
        'nature': ['natural', 'environmental', 'ambient'],
        'mechanical': ['mechanical', 'industrial', 'machine'],
        'voice': ['vocal', 'speech', 'talking', 'speaking'],
        'impact': ['impact', 'hit', 'crash', 'bang'],
        'movement': ['movement', 'motion', 'action']
    }

    enhanced = text_prompt.strip()

    # Add audio context if not already present
    if not any(keyword in enhanced.lower() for keywords in audio_keywords.values() for keyword in keywords):
        if any(word in enhanced.lower() for word in ['music', 'song', 'melody', 'instrument']):
            enhanced = f"musical {enhanced}"
        elif any(word in enhanced.lower() for word in ['voice', 'talk', 'speak', 'say']):
            enhanced = f"vocal {enhanced}"
        elif any(word in enhanced.lower() for word in ['water', 'wind', 'rain', 'nature']):
            enhanced = f"natural environmental {enhanced}"
        else:
            enhanced = f"audio of {enhanced}"

    # Ensure it ends with audio context
    if not enhanced.lower().endswith(('sound', 'sounds', 'audio', 'noise')):
        enhanced = f"{enhanced} sound"

    return enhanced

def get_audio_prompt_templates() -> Dict[str, str]:
    """
    Get predefined prompt templates for common audio generation scenarios.

    Returns:
        Dictionary of template categories and prompts
    """
    return {
        "music": {
            "ambient": "ambient atmospheric music, soft melodic tones",
            "upbeat": "upbeat energetic music, rhythmic and lively",
            "dramatic": "dramatic cinematic music, intense and emotional",
            "peaceful": "peaceful calm music, gentle and soothing"
        },
        "nature": {
            "forest": "natural forest sounds, birds chirping, leaves rustling",
            "ocean": "ocean waves, water sounds, peaceful seaside ambience",
            "rain": "gentle rain sounds, water droplets, calming precipitation",
            "wind": "wind sounds, air movement, natural atmospheric audio"
        },
        "urban": {
            "traffic": "urban traffic sounds, cars passing, city ambience",
            "crowd": "crowd sounds, people talking, social atmosphere",
            "construction": "construction sounds, building work, industrial activity",
            "cafe": "cafe ambience, coffee shop sounds, social chatter"
        },
        "action": {
            "footsteps": "footsteps walking, movement sounds, human activity",
            "running": "running sounds, fast movement, energetic motion",
            "impact": "impact sounds, hitting, collision effects",
            "mechanical": "mechanical sounds, machine operation, industrial audio"
        }
    }

def create_safe_negative_conditioning(positive_conditioning: Dict, negative_prompt: str, device: str = "cpu") -> Dict:
    """
    Create negative conditioning that matches the tensor dimensions of positive conditioning.

    Args:
        positive_conditioning: The positive conditioning dictionary
        negative_prompt: The negative text prompt
        device: Device to create tensors on

    Returns:
        Negative conditioning dictionary with matching tensor dimensions
    """
    try:
        # Create a copy of positive conditioning structure
        negative_conditioning = positive_conditioning.copy()

        # Replace only the text prompt, keep all other tensors the same
        negative_conditioning["text_prompt"] = negative_prompt

        print(f"AudioX: Created safe negative conditioning with prompt: '{negative_prompt}'")
        return negative_conditioning

    except Exception as e:
        print(f"AudioX: Failed to create safe negative conditioning: {e}")
        return None

def analyze_prompt_specificity(prompt: str) -> float:
    """
    Analyze how specific and detailed a prompt is to determine optimal CFG scaling.

    Args:
        prompt: Text prompt to analyze

    Returns:
        Specificity score from 0.0 (very general) to 1.0 (very specific)
    """
    if not prompt or len(prompt.strip()) == 0:
        return 0.0

    # Factors that indicate specificity
    specificity_indicators = {
        'descriptive_words': ['clear', 'soft', 'loud', 'gentle', 'harsh', 'smooth', 'rough', 'deep', 'high', 'low'],
        'technical_terms': ['reverb', 'echo', 'distortion', 'compression', 'eq', 'frequency', 'amplitude', 'stereo', 'mono'],
        'instruments': ['piano', 'guitar', 'violin', 'drums', 'bass', 'synthesizer', 'flute', 'trumpet', 'saxophone'],
        'genres': ['jazz', 'classical', 'rock', 'electronic', 'ambient', 'folk', 'blues', 'reggae', 'hip-hop'],
        'emotions': ['happy', 'sad', 'angry', 'peaceful', 'energetic', 'melancholic', 'joyful', 'dramatic'],
        'environments': ['forest', 'ocean', 'city', 'room', 'hall', 'studio', 'outdoor', 'indoor', 'cave'],
        'actions': ['walking', 'running', 'typing', 'cooking', 'driving', 'flying', 'swimming', 'dancing']
    }

    prompt_lower = prompt.lower()
    word_count = len(prompt.split())
    specificity_score = 0.0

    # Base score from word count (longer prompts tend to be more specific)
    specificity_score += min(word_count / 20.0, 0.3)  # Max 0.3 from length

    # Score from specific indicators
    total_indicators = sum(len(indicators) for indicators in specificity_indicators.values())
    found_indicators = 0

    for category, indicators in specificity_indicators.items():
        for indicator in indicators:
            if indicator in prompt_lower:
                found_indicators += 1
                # Weight certain categories more
                if category in ['technical_terms', 'instruments']:
                    specificity_score += 0.1
                else:
                    specificity_score += 0.05

    # Bonus for multiple descriptive adjectives
    adjectives = ['beautiful', 'amazing', 'perfect', 'excellent', 'wonderful', 'terrible', 'awful', 'great']
    adj_count = sum(1 for adj in adjectives if adj in prompt_lower)
    specificity_score += min(adj_count * 0.05, 0.15)

    # Bonus for numbers and specific measurements
    import re
    numbers = re.findall(r'\d+', prompt)
    if numbers:
        specificity_score += min(len(numbers) * 0.05, 0.1)

    return min(specificity_score, 1.0)

def calculate_adaptive_cfg(base_cfg: float, prompt: str, prompt_type: str = "audio") -> float:
    """
    Calculate adaptive CFG scale based on prompt specificity and type.

    Args:
        base_cfg: Base CFG scale
        prompt: Text prompt to analyze
        prompt_type: Type of prompt ("audio", "music", "video")

    Returns:
        Optimized CFG scale
    """
    specificity = analyze_prompt_specificity(prompt)

    # Adjust CFG based on specificity
    if specificity > 0.7:  # Very specific prompts
        cfg_multiplier = 1.3  # Higher CFG for specific prompts
    elif specificity > 0.4:  # Moderately specific
        cfg_multiplier = 1.1
    else:  # General prompts
        cfg_multiplier = 0.9  # Lower CFG for general prompts

    # Type-specific adjustments
    type_adjustments = {
        "music": 1.0,  # Music often needs precise control
        "audio": 1.1,  # Audio effects benefit from higher CFG
        "video": 0.95  # Video conditioning is already strong
    }

    cfg_multiplier *= type_adjustments.get(prompt_type, 1.0)

    adaptive_cfg = base_cfg * cfg_multiplier

    # Clamp to reasonable range
    adaptive_cfg = max(3.0, min(adaptive_cfg, 15.0))

    print(f"AudioX: Adaptive CFG - Specificity: {specificity:.2f}, Base: {base_cfg}, Adaptive: {adaptive_cfg:.2f}")
    return adaptive_cfg

def expand_audio_keywords(prompt: str) -> str:
    """
    Expand audio-related keywords with synonyms and related terms for better conditioning.

    Args:
        prompt: Original prompt

    Returns:
        Expanded prompt with additional related terms
    """
    # Audio keyword expansions
    expansions = {
        'footsteps': 'footsteps walking steps movement',
        'music': 'music musical melody harmony',
        'piano': 'piano keyboard keys musical',
        'guitar': 'guitar strings musical instrument',
        'drums': 'drums percussion beats rhythm',
        'water': 'water liquid flowing aquatic',
        'wind': 'wind air breeze atmospheric',
        'rain': 'rain precipitation water droplets',
        'fire': 'fire crackling burning flames',
        'car': 'car vehicle automobile engine',
        'voice': 'voice vocal speech human',
        'bird': 'bird avian chirping singing',
        'ocean': 'ocean sea waves water',
        'forest': 'forest trees nature woodland',
        'city': 'city urban metropolitan traffic',
        'thunder': 'thunder storm lightning rumble',
        'bell': 'bell chime ringing metallic',
        'clock': 'clock ticking time mechanical'
    }

    expanded_prompt = prompt
    prompt_lower = prompt.lower()

    for keyword, expansion in expansions.items():
        if keyword in prompt_lower and expansion not in prompt_lower:
            # Add expansion terms that aren't already present
            new_terms = [term for term in expansion.split() if term not in prompt_lower and term != keyword]
            if new_terms:
                expanded_prompt += f", {', '.join(new_terms[:2])}"  # Add max 2 new terms

    return expanded_prompt

def emphasize_key_terms(prompt: str) -> str:
    """
    Emphasize important audio terms by strategic repetition and positioning.

    Args:
        prompt: Original prompt

    Returns:
        Prompt with emphasized key terms
    """
    # Terms that benefit from emphasis
    emphasis_terms = {
        'clear': 'clear, well-defined',
        'soft': 'soft, gentle',
        'loud': 'loud, powerful',
        'fast': 'fast, rapid',
        'slow': 'slow, gradual',
        'deep': 'deep, low-frequency',
        'high': 'high, treble',
        'smooth': 'smooth, flowing',
        'rough': 'rough, textured',
        'peaceful': 'peaceful, calm',
        'energetic': 'energetic, dynamic',
        'dramatic': 'dramatic, intense'
    }

    emphasized_prompt = prompt
    prompt_lower = prompt.lower()

    for term, emphasis in emphasis_terms.items():
        if term in prompt_lower and emphasis not in prompt_lower:
            emphasized_prompt = emphasized_prompt.replace(term, emphasis, 1)

    return emphasized_prompt

def create_multi_aspect_conditioning(prompt: str, duration_seconds: int, device: str = "cpu",
                                   conditioning_strength: float = 1.0) -> list:
    """
    Create multiple conditioning vectors from different aspects of the prompt for more robust conditioning.

    Args:
        prompt: Text prompt
        duration_seconds: Duration for conditioning
        device: Device to create tensors on
        conditioning_strength: Strength multiplier for conditioning

    Returns:
        List of conditioning dictionaries
    """
    # Break down prompt into different aspects
    aspects = []

    # Main prompt
    aspects.append(prompt)

    # Enhanced version with keywords
    enhanced = expand_audio_keywords(prompt)
    if enhanced != prompt:
        aspects.append(enhanced)

    # Emphasized version
    emphasized = emphasize_key_terms(prompt)
    if emphasized != prompt and emphasized != enhanced:
        aspects.append(emphasized)

    # Create conditioning for each aspect
    conditioning_list = []
    for aspect in aspects[:2]:  # Limit to 2 aspects to avoid overcomplication
        try:
            conditioning = create_minimal_conditioning(aspect, duration_seconds, device)
            conditioning_list.append(conditioning)
        except Exception as e:
            print(f"AudioX: Failed to create conditioning for aspect '{aspect}': {e}")

    # If we couldn't create multiple aspects, return at least the main one
    if not conditioning_list:
        conditioning_list = [create_minimal_conditioning(prompt, duration_seconds, device)]

    print(f"AudioX: Created {len(conditioning_list)} conditioning aspects")
    return conditioning_list

def create_super_enhanced_prompt(prompt: str, prompt_type: str = "audio",
                               style_modifier: str = "none", add_quality: bool = True) -> str:
    """
    Create a super-enhanced prompt using all available enhancement techniques.

    Args:
        prompt: Original prompt
        prompt_type: Type of prompt ("audio", "music")
        style_modifier: Style to apply
        add_quality: Whether to add quality terms

    Returns:
        Super-enhanced prompt
    """
    if not prompt or len(prompt.strip()) == 0:
        return "high quality audio" if prompt_type == "audio" else "high quality music"

    enhanced = prompt.strip()

    # Step 1: Apply style modifier first
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
            enhanced = f"{style_terms[style_modifier]} {enhanced}"

    # Step 2: Expand keywords
    enhanced = expand_audio_keywords(enhanced)

    # Step 3: Emphasize key terms
    enhanced = emphasize_key_terms(enhanced)

    # Step 4: Add audio/music context if missing
    enhanced = enhance_audio_prompt(enhanced)

    # Step 5: Add quality terms
    if add_quality:
        quality_terms = "high quality, clear, well-defined"
        if quality_terms not in enhanced.lower():
            enhanced = f"{enhanced}, {quality_terms}"

    # Step 6: Add type-specific enhancements
    if prompt_type == "music":
        if "music" not in enhanced.lower():
            enhanced = f"musical {enhanced}"
        # Add musical context
        if not any(term in enhanced.lower() for term in ['melody', 'harmony', 'rhythm', 'musical']):
            enhanced = f"{enhanced}, melodic"

    # Step 7: Strategic repetition of key terms for emphasis
    key_words = []
    for word in enhanced.split():
        word_clean = word.strip('.,!?').lower()
        if word_clean in ['clear', 'high', 'quality', 'musical', 'realistic', 'natural']:
            if word_clean not in [w.lower() for w in key_words]:
                key_words.append(word_clean)

    if key_words and len(key_words) <= 2:  # Only repeat if we have 1-2 key terms
        enhanced = f"{enhanced}, {', '.join(key_words)}"

    print(f"AudioX: Super-enhanced prompt: '{enhanced}'")
    return enhanced

def create_empty_conditioning_values(device: str = "cpu", duration_seconds: int = 10) -> Dict:
    """
    Create appropriate empty values for conditioning that conditioners can handle.
    Only use this when you need to provide empty values for multi-modal conditioners.

    Args:
        device: Device to create tensors on
        duration_seconds: Duration for audio conditioning (affects audio tensor size)

    Returns:
        Dictionary with appropriate empty values for each conditioning type
    """
    import torch

    # Video conditioner expects: duration=10s, fps=5, so 50 frames total
    # Shape: (batch, time_length, channels, height, width) = (1, 50, 3, 224, 224)
    video_frames = 50  # 10 seconds * 5 fps

    # Audio conditioner expects audio tensor with shape (batch, channels, samples)
    # For empty audio, create a zero tensor with appropriate sample rate
    sample_rate = 44100  # Standard sample rate
    audio_samples = int(duration_seconds * sample_rate)

    return {
        "video_prompt": [torch.zeros(1, video_frames, 3, 224, 224, device=device)],  # Empty video tensor with correct frame count
        "audio_prompt": [torch.zeros(1, 2, audio_samples, device=device)],  # Empty stereo audio tensor
    }

def clear_model_cache():
    """Clear the model cache to free memory."""
    global _model_cache
    for key, (model, _) in _model_cache.items():
        del model
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("AudioX model cache cleared")


