from PIL import Image
import math
import numpy as np
import torch
import os

def _find_best_patch_grid(target_patches, aspect_ratio):
    """Find the optimal patch grid dimensions that best match the aspect ratio."""
    best_height_patches = 1
    best_width_patches = target_patches
    min_ratio_diff = float('inf')
    
    for i in range(1, int(math.sqrt(target_patches)) + 1):
        if target_patches % i == 0:
            height_patches = i
            width_patches = target_patches // i
            target_ratio = width_patches / height_patches
            ratio_diff = abs(target_ratio - aspect_ratio)
            
            if ratio_diff < min_ratio_diff:
                min_ratio_diff = ratio_diff
                best_height_patches = height_patches
                best_width_patches = width_patches
    
    return best_height_patches, best_width_patches

def _resize_with_padding(img, target_width, target_height):
    """Resize image with black padding and centering."""
    orig_width, orig_height = img.size
    
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height
    scale = min(scale_w, scale_h)
    
    resized_width = min(int(round(orig_width * scale)), target_width)
    resized_height = min(int(round(orig_height * scale)), target_height)
    
    resized_img = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    
    new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    new_img.paste(resized_img, (offset_x, offset_y))
    
    return new_img

def _resize_simple(img, max_pixel_area):
    """Simple resize that only scales down, preserving aspect ratio."""
    orig_width, orig_height = img.size
    current_pixel_area = orig_width * orig_height
    
    scale = min(math.sqrt(max_pixel_area / current_pixel_area), 1.0)
    new_width = int(round(orig_width * scale))
    new_height = int(round(orig_height * scale))
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def resize_audio_for_npu(audios, npu_static_length=163839):
    """Process audio data by padding or trimming to match the target length."""
    if audios is None:
        return None
    
    if not isinstance(audios, list) or len(audios) != 1:
        raise ValueError("Only one audio supported for NPU. Please set device to GPU")
    
    audio = audios[0]
    
    if len(audio) > npu_static_length:
        print(f"Warning: Audio data {len(audio)} exceeds the target length {npu_static_length}. It will be trimmed to the target size.")
        return [audio[:npu_static_length]]
    
    pad_length = npu_static_length - len(audio)
    padded_audio = np.concatenate([audio, np.zeros(pad_length, dtype=audio.dtype)])
    print(f"Warning: Audio data {len(audio)} smaller than the target length {npu_static_length}. It will be padding to the target size.")
    return [padded_audio]

def resize_image_for_npu(imgs, patch_size=14, npu_static_patch_length=2048):
    """Resize images for NPU processing with fixed patch count and black padding."""
    if imgs is None:
        return None

    if not isinstance(imgs, list) or len(imgs) == 0:
        raise ValueError("Input imgs must be a non-empty list.")

    resized_imgs = []
    
    for img in imgs:
        orig_width, orig_height = img.size
        aspect_ratio = orig_width / orig_height
        
        best_height_patches, best_width_patches = _find_best_patch_grid(npu_static_patch_length, aspect_ratio)
        
        target_height = best_height_patches * patch_size
        target_width = best_width_patches * patch_size
        
        new_img = _resize_with_padding(img, target_width, target_height)
        
        print(f"Resize image: {orig_width}x{orig_height} -> {target_width}x{target_height} with patch number {npu_static_patch_length}")
        resized_imgs.append(new_img)

    return resized_imgs

def resize_images_for_cpu_gpu(images, patch_size=14, target_patch_size_each_img=2048):
    """Resize images for CPU/GPU processing with flexible patch count."""
    if images is None:
        return None
    
    if not isinstance(images, list) or len(images) == 0:
        raise ValueError("Input images must be a non-empty list.")
    
    processed_images = []
    max_pixel_area = target_patch_size_each_img * (patch_size ** 2)
    
    for img in images:
        orig_width, orig_height = img.size
        resized_img = _resize_simple(img, max_pixel_area)
        new_width, new_height = resized_img.size
        
        processed_images.append(np.array(resized_img))
        
        actual_patch_count = round(new_width * new_height / (patch_size ** 2))
        print(f"Resize image: {orig_width}x{orig_height} -> {new_width}x{new_height} with patch number {actual_patch_count}")

    return processed_images

def resize_videos_for_cpu_gpu(video_list, patch_size=14, target_patch_size_each_frame=2048):
    """Resize video for CPU/GPU processing with flexible patch count per frame."""
    if video_list is None or len(video_list) == 0:
        return None
    
    if not isinstance(video_list, list) or len(video_list) != 1:
        raise ValueError("Input video_list must contain exactly one video tensor.")
    
    video = video_list[0]
    frames, channels, orig_height, orig_width = video.shape
    max_pixel_area = target_patch_size_each_frame * (patch_size ** 2)
    
    processed_frames = []
    
    for i in range(frames):
        frame = video[i].permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(frame)
        
        resized_img = _resize_simple(img, max_pixel_area)
        new_width, new_height = resized_img.size
        
        processed_frames.append(torch.from_numpy(np.array(resized_img)).permute(2, 0, 1))
        
        actual_patch_count = round(new_width * new_height / (patch_size ** 2))
        print(f"Resize frame {i+1}/{frames}: {orig_width}x{orig_height} -> {new_width}x{new_height} with patch number {actual_patch_count}")
    
    return [torch.stack(processed_frames, dim=0)]

def resize_videos_for_npu(video_list, patch_size=14, target_patch_size_each_frame=2048):
    """Resize video for NPU processing with fixed patch count and black padding."""
    if video_list is None or len(video_list) == 0:
        return None
    
    video = video_list[0]
    frames, channels, orig_height, orig_width = video.shape
    aspect_ratio = orig_width / orig_height
    
    best_height_patches, best_width_patches = _find_best_patch_grid(target_patch_size_each_frame, aspect_ratio)
    
    target_height = best_height_patches * patch_size
    target_width = best_width_patches * patch_size
    
    processed_frames = []
    os.makedirs('inputs/video_imgs', exist_ok=True)
    
    for i in range(frames):
        frame = video[i].permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(frame)
        
        new_img = _resize_with_padding(img, target_width, target_height)
        
        processed_frames.append(torch.from_numpy(np.array(new_img)).permute(2, 0, 1))
        
        frame_filename = os.path.join('inputs/video_imgs', f"frame_{i+1:04d}.png")
        new_img.save(frame_filename)

        print(f"Resize frame {i+1}/{frames}: {orig_width}x{orig_height} -> {target_width}x{target_height} with patch number {target_patch_size_each_frame}")
    
    return [torch.stack(processed_frames, dim=0)]

def resize_inputs(audios, images, videos, audio_len, img_patch_size, patch_length_per_img, device):
    """Main entry point for resizing all input types based on device."""
    if device == "NPU":
        audios = resize_audio_for_npu(audios, npu_static_length=audio_len)
        images = resize_image_for_npu(images, patch_size=img_patch_size, npu_static_patch_length=patch_length_per_img)
        videos = resize_videos_for_npu(videos, patch_size=img_patch_size, target_patch_size_each_frame=patch_length_per_img)
    else:  # CPU or GPU
        images = resize_images_for_cpu_gpu(images, patch_size=img_patch_size, target_patch_size_each_img=patch_length_per_img)
        videos = resize_videos_for_cpu_gpu(videos, patch_size=img_patch_size, target_patch_size_each_frame=patch_length_per_img)

    return audios, images, videos

def dump_inputs_info(inputs):
    """Print detailed information about processed inputs."""
    print("=== Inputs Informations ===")
    print(f"key values: {inputs.keys()}")
    
    for key, value in inputs.items():
        print(f"\n{key} shape: {value.shape}")
        if hasattr(value, 'dtype'):
            print(f"{key} type: {value.dtype}")
        print(f"{key} value: {value}")

    if 'input_ids' in inputs:
        print(f"Total token length for Thinker LLM: {inputs['input_ids'].size(1)}")
    if 'pixel_values' in inputs:
        print(f" - Vision Embedding input length: {inputs['pixel_values'].size(0)}, output token length: {inputs['pixel_values'].size(0)/4}")
    if 'image_grid_thw' in inputs:
        print(f" - Vision Embedding include: {inputs['image_grid_thw'].size(0)} images")
    if 'feature_attention_mask' in inputs:
        print(f" - Audio Embedding input length: {inputs['feature_attention_mask'].sum(-1)}, output token length: {inputs['feature_attention_mask'].sum(-1)/4}")


# Dataset helpers for calibration
import nncf
from typing import Optional

def check_wikitext2_availability():
    """Check if wikitext2 dataset is available locally or can be downloaded."""
    try:
        from datasets import load_dataset
        return True
    except ImportError:
        print("⚠️ 'datasets' library not found. Install with: pip install datasets")
        return False

def get_wikitext2_dataset(model_id: str, num_samples: int = 128) -> Optional[nncf.Dataset]:
    """
    Get wikitext2 dataset for calibration.
    
    Args:
        model_id: Model identifier for tokenizer
        num_samples: Number of samples to use for calibration
        
    Returns:
        NNCF Dataset object or None if unavailable
    """
    if not check_wikitext2_availability():
        return None
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # Try to load wikitext2 dataset
        print("⌛ Downloading/loading wikitext2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def prepare_input(example):
            """Prepare input for the model."""
            text = example["text"]
            if not text or len(text.strip()) < 10:  # Skip empty or very short texts
                return None
            
            # Tokenize text
            inputs = tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Return inputs in the format expected by the model
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            }
        
        def data_generator():
            """Generate calibration data."""
            count = 0
            for example in dataset:
                if count >= num_samples:
                    break
                    
                prepared = prepare_input(example)
                if prepared is not None:
                    # For embedding layer, we only need input_ids
                    yield prepared["input_ids"].unsqueeze(0)  # Add batch dimension
                    count += 1
        
        # Create NNCF dataset
        calibration_dataset = nncf.Dataset(data_generator)
        print(f"✅ Created calibration dataset with {num_samples} samples")
        
        return calibration_dataset
        
    except Exception as e:
        print(f"❌ Error loading wikitext2 dataset: {e}")
        
        # Try to use a simple text dataset as fallback
        print("⌛ Creating fallback calibration dataset...")
        return create_fallback_dataset(model_id, num_samples)

def create_fallback_dataset(model_id: str, num_samples: int = 128) -> Optional[nncf.Dataset]:
    """
    Create a simple fallback calibration dataset using predefined texts.
    
    Args:
        model_id: Model identifier for tokenizer
        num_samples: Number of samples to generate
        
    Returns:
        NNCF Dataset object or None
    """
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Sample texts for calibration
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Computer vision helps machines interpret and understand visual information.",
            "Artificial intelligence is transforming various industries worldwide.",
            "Neural networks are inspired by the structure of the human brain.",
            "Data science combines statistics, computer science, and domain expertise.",
            "Cloud computing provides on-demand access to computing resources.",
            "Cybersecurity protects digital systems from malicious attacks.",
        ]
        
        def data_generator():
            """Generate calibration data from sample texts."""
            for i in range(num_samples):
                text = sample_texts[i % len(sample_texts)]
                
                inputs = tokenizer(
                    text,
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                yield inputs["input_ids"].squeeze(0).unsqueeze(0)  # Add batch dimension
        
        calibration_dataset = nncf.Dataset(data_generator)
        print(f"✅ Created fallback calibration dataset with {num_samples} samples")
        
        return calibration_dataset
        
    except Exception as e:
        print(f"❌ Error creating fallback dataset: {e}")
        return None