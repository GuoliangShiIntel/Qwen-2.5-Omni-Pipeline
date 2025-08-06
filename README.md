# Qwen 2.5 Omni 7B Model Pipeline

This repository contains the pipeline for the Qwen 2.5 Omni 7B model.

## Model Information

- Hugging Face: [Qwen 2.5 Omni 7B](https://hf-mirror.com/Qwen/Qwen2.5-Omni-7B)

## Setup Environment

1. Create a virtual environment:

    ```bash
    python3 -m venv omin_env
    ```

2. Activate the virtual environment:

    - On Unix or MacOS:

        ```bash
        source omin_env/bin/activate
        ```

    - On Windows:

        ```bash
        omin_env\Scripts\activate
        ```

3. Install the required packages:

    ```bash
    pip install "transformers==4.52.0" "torchvision" "accelerate" "qwen-omni-utils[decord]" "gradio>=4.19" --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu
    pip install "openvino==2025.1.0" "nncf>=2.16.0"
    ```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/GuoliangShiIntel/Qwen-2.5-Omni-Pipeline.git
    ```

2. Convert the model to OpenVINO IR format:

    ```bash
    cd Qwen-2.5-Omni-Pipeline
    python convert_to_ov_model.py
    ```

3. Run the demo:

    ```bash
    python omin_demo.py
    ```

## Demo Use Cases

This repository provides two main demo applications to showcase different capabilities of the Qwen 2.5 Omni model:

### 1. Basic Multimodal Demo (`omin_demo.py`)

This is a basic demonstration that processes pre-configured multimodal inputs including text, images, and audio.

**Features:**
- Text generation from multimodal inputs (images + audio + text prompts)
- Support for both Chinese and English text processing
- Optional speech synthesis output (when `enable_talker=True`)
- Configurable device allocation (NPU/GPU/CPU) for different model components

**Usage:**
```bash
python omin_demo.py
```

**Configuration:**
- Edit the `conversation` variable in the script to customize input prompts and media files
- Modify device settings (`thinker_device`, `talker_device`, `token2wav_device`) based on your hardware
- Set `enable_talker=True` to generate speech output in addition to text

**Sample Input:**
- Text: "你从图片里面看到了什么？" (What do you see in the image?)
- Image: User behavior screenshots
- Audio: Sample audio files (e.g., Trailer.wav)

### 2. Interactive Screen Capture Demo (`omni_demo_screen.py`)

This is an advanced interactive demo that allows real-time screen capture and video recording for multimodal analysis.

**Features:**
- Real-time screen capture and analysis
- Streaming video recording with frame-by-frame processing
- Interactive keyboard controls for different capture modes
- CDPruner integration for efficient video token compression
- Frame selection optimization (processes only selected frames for efficiency)

**Usage:**
```bash
python omni_demo_screen.py
```

**Interactive Controls:**
- **[SPACE]**: Take a screenshot and analyze the current screen content
- **[V]**: Start streaming video recording and real-time analysis
- **[ESC]**: Exit the application

**Key Features:**
- **Screen Capture**: Captures the current screen and asks the model to describe what it sees
- **Video Streaming**: Records screen activity and processes it in real-time with frame selection
- **CDPruner**: Reduces computational overhead by compressing visual tokens (enabled by default)
- **Frame Selection**: Processes only specific frames (2nd and 4th by default) for better performance

**Configuration Options:**
- `frame_selection_enabled`: Enable/disable selective frame processing
- `selected_frame_indices`: Choose which frames to process from video segments
- `cdpruner_num_visual_tokens`: Control the number of visual tokens for compression
- Device allocation for different model components

**Use Cases:**
- Screen content analysis and description
- Real-time user behavior monitoring
- Interactive screen-based Q&A sessions
- Screen recording with AI-powered commentary

## Model Variants

The demos support multiple model variants located in the following directories:
- `Qwen2.5-Omni-3B-INT4-SYM/`: 3B model with INT4 symmetric quantization
- `Qwen2.5-Omni-3B-NF4/`: 3B model with NF4 quantization
- `Qwen2.5-Omni-7B-INT4-SYM/`: 7B model with INT4 symmetric quantization  
- `Qwen2.5-Omni-7B-NF4/`: 7B model with NF4 quantization

To switch between models, modify the `model_id` variable in the demo scripts.

Feel free to reach out if you have any questions or need further assistance.
