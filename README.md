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
    pip install "transformers>=4.52.0" "torchvision" "accelerate" "qwen-omni-utils[decord]" "gradio>=4.19" --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu
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

Feel free to reach out if you have any questions or need further assistance.
