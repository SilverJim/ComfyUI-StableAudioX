# Enhanced AudioX Generation System

This document describes the comprehensive improvements made to the ComfyUI-AudioX generation system to enhance prompt adherence and provide better control over text-to-audio, text-to-music, and video-to-audio generation.

## Issues Identified

The original video-to-audio implementation had several limitations:

1. **Weak Text-Video Balance**: Text and video conditioning were processed separately without proper balancing
2. **Fixed CFG Scale**: Single CFG scale didn't allow fine-tuning text vs video influence  
3. **No Negative Prompting**: Missing negative conditioning capabilities
4. **Basic Text Processing**: No audio-domain specific text preprocessing
5. **Limited Conditioning Control**: No user control over conditioning weights

## New Features

### 1. AudioXEnhancedTextToAudio Node

An advanced text-to-audio generation node with enhanced prompt processing:

#### Advanced Prompting Features
- **Negative Prompting**: Specify audio characteristics to avoid (e.g., "muffled, distorted, low quality")
- **Prompt Templates**: Pre-defined templates for common audio scenarios
- **Style Modifiers**: cinematic, realistic, ambient, dramatic, peaceful, energetic
- **Auto Enhancement**: Automatically adds audio-specific keywords

#### Enhanced Controls
- **Higher CFG Scale Range**: Up to 20.0 for stronger prompt adherence
- **Template Integration**: Seamlessly combines base prompts with templates
- **Quality Enhancement**: Automatic addition of quality terms

#### Advanced Conditioning Features
- **Conditioning Modes**:
  - `standard`: Basic conditioning
  - `enhanced`: Audio-specific keyword enhancement
  - `super_enhanced`: All enhancement techniques combined
  - `multi_aspect`: Multiple conditioning vectors for robust generation
- **Adaptive CFG**: Automatically adjusts CFG scale based on prompt specificity
- **Keyword Expansion**: Adds related audio terms and synonyms
- **Term Emphasis**: Strategic repetition and emphasis of key terms
- **Context-Aware Processing**: Understands prompt intent and enhances accordingly

### 2. AudioXEnhancedTextToMusic Node

A specialized music generation node with musical attributes:

#### Musical Style Controls
- **Music Styles**: classical, jazz, electronic, ambient, rock, folk, cinematic
- **Tempo Control**: slow, moderate, fast, very_fast
- **Mood Settings**: happy, sad, peaceful, energetic, mysterious, dramatic
- **Negative Prompting**: Avoid discordant, harsh, or atonal characteristics

#### Smart Music Enhancement
- **Automatic Music Context**: Ensures prompts are interpreted as musical
- **Style Integration**: Combines multiple musical attributes intelligently
- **Enhanced Descriptions**: Adds appropriate musical terminology

### 3. AudioXEnhancedVideoToAudio Node

An advanced video-to-audio generation node with the following improvements:

#### Separate CFG Controls
- **Text CFG Scale**: Independent control over text conditioning strength (default: 7.0)
- **Video CFG Scale**: Independent control over video conditioning strength (default: 7.0)
- **Effective CFG**: Automatically calculated weighted average based on conditioning weights

#### Conditioning Weight Controls
- **Text Weight**: Control the influence of text conditioning (0.0-2.0, default: 1.0)
- **Video Weight**: Control the influence of video conditioning (0.0-2.0, default: 1.0)
- **Balanced Generation**: Allows fine-tuning the balance between following text prompts vs video content

#### Advanced Prompting
- **Negative Prompting**: Specify what audio characteristics to avoid
- **Prompt Templates**: Pre-defined templates for common audio scenarios
- **Auto Enhancement**: Automatically add audio-specific keywords to improve generation

### 4. AudioXPromptHelper Node

A utility node for creating better audio prompts:

#### Template Categories
- **Music**: ambient, upbeat, dramatic, peaceful
- **Nature**: forest, ocean, rain, wind  
- **Urban**: traffic, crowd, construction, cafe
- **Action**: footsteps, running, impact, mechanical

#### Enhancement Features
- **Auto Enhancement**: Adds audio-specific context to prompts
- **Quality Terms**: Adds quality enhancement terms like "high quality, clear"
- **Style Modifiers**: cinematic, realistic, ambient, dramatic, peaceful, energetic
- **Negative Prompts**: Suggests appropriate negative terms

### 5. Enhanced Conditioning Pipeline

#### Audio-Specific Text Processing
```python
def enhance_audio_prompt(text_prompt: str) -> str:
    """Enhance text prompt for better audio generation"""
    # Adds audio context if missing
    # Emphasizes audio-specific keywords
    # Ensures proper audio terminology
```

#### Better Text-Video Fusion
```python
def create_enhanced_video_conditioning(video_tensor, text_prompt, 
                                     text_weight=1.0, video_weight=1.0,
                                     negative_prompt=""):
    """Create enhanced conditioning with better text-video balance"""
    # Processes text with audio-specific enhancements
    # Applies conditioning weights
    # Includes negative prompting support
```

## Usage Examples

### Enhanced Text-to-Audio Generation
```json
{
  "text_prompt": "footsteps on wooden floor",
  "cfg_scale": 8.0,
  "negative_prompt": "muffled, distorted, low quality",
  "prompt_template": "action_footsteps",
  "style_modifier": "realistic",
  "enhance_prompt": true
}
```

### Enhanced Text-to-Music Generation
```json
{
  "text_prompt": "peaceful piano melody",
  "cfg_scale": 7.0,
  "negative_prompt": "discordant, harsh, atonal",
  "music_style": "classical",
  "tempo": "slow",
  "mood": "peaceful",
  "enhance_prompt": true
}
```

### Enhanced Video-to-Audio Generation
```json
{
  "text_prompt": "footsteps on wooden floor",
  "text_cfg_scale": 8.0,
  "video_cfg_scale": 6.0,
  "text_weight": 1.2,
  "video_weight": 0.8,
  "negative_prompt": "muffled, distorted, low quality"
}
```

### Using Prompt Templates
```json
{
  "base_prompt": "person walking",
  "template": "action_footsteps",
  "style_modifier": "realistic",
  "enhance_automatically": true
}
```

## Best Practices

### For Better Prompt Adherence

1. **Use Specific Audio Descriptions**
   - Instead of: "person walking"
   - Use: "clear footsteps on wooden floor, steady rhythm"

2. **Balance Text and Video Weights**
   - High text weight (1.5-2.0) for specific audio requirements
   - High video weight (1.5-2.0) for video-synchronized audio
   - Balanced weights (1.0 each) for general video-to-audio

3. **Leverage Negative Prompting**
   - Always include: "muffled, distorted, low quality"
   - Add specific negatives: "echo, reverb" for dry sounds
   - Use "silence, quiet" to avoid empty audio

4. **Adjust CFG Scales**
   - Higher text CFG (8.0-12.0) for strong prompt adherence
   - Lower video CFG (4.0-6.0) when text is very specific
   - Balanced CFG (7.0 each) for general use

### Prompt Templates Guide

#### Music Generation
- **ambient**: "ambient atmospheric music, soft melodic tones"
- **upbeat**: "upbeat energetic music, rhythmic and lively"
- **dramatic**: "dramatic cinematic music, intense and emotional"

#### Nature Sounds
- **forest**: "natural forest sounds, birds chirping, leaves rustling"
- **ocean**: "ocean waves, water sounds, peaceful seaside ambience"
- **rain**: "gentle rain sounds, water droplets, calming precipitation"

#### Action Sounds
- **footsteps**: "footsteps walking, movement sounds, human activity"
- **impact**: "impact sounds, hitting, collision effects"
- **mechanical**: "mechanical sounds, machine operation, industrial audio"

## Technical Implementation

### Conditioning Flow
1. **Text Enhancement**: Audio-specific keyword processing
2. **Template Application**: Pre-defined prompt templates
3. **Weight Application**: Balanced text-video conditioning
4. **CFG Calculation**: Weighted average of separate CFG scales
5. **Generation**: Enhanced conditioning pipeline

### Key Functions
- `enhance_audio_prompt()`: Audio-specific text processing
- `create_enhanced_video_conditioning()`: Advanced conditioning creation
- `get_audio_prompt_templates()`: Template management

## Workflow Examples

See the included workflow files:
- `examples/enhanced_video_to_audio_workflow.json`: Complete enhanced workflow
- `examples/simple_video_to_audio_workflow.json`: Basic workflow for comparison

## Performance Notes

- Enhanced conditioning adds minimal computational overhead
- Text processing is lightweight and fast
- CFG calculation is optimized for real-time adjustment
- Template system provides instant prompt improvements

## Future Improvements

Potential areas for further enhancement:
1. **Cross-Modal Attention**: Direct attention between text and video features
2. **Audio-Aware Text Encoding**: CLAP-based text encoding for better audio alignment
3. **Hierarchical Conditioning**: Separate global and local audio descriptions
4. **Adaptive CFG**: Dynamic CFG adjustment based on prompt specificity
5. **Real-time Preview**: Quick audio previews during parameter adjustment
