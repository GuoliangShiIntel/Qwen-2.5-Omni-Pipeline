from PIL import Image
import math
import numpy as np

def resize_audio_for_npu(audios, npu_static_length=163839):
    """
    Process audio data by padding or trimming to match the target length.

    Args:
        audios (list | None): Input audio data.
        npu_static_length (int): The static length for NPU processing.

    Returns:
        list | None: Processed audio data, or None if input is None.
    """
    if audios is None:
        return None
    
    if not isinstance(audios, list) or len(audios) != 1:
        raise ValueError("Only one audio supported for NPU. Please set device to GPU")
    
    audio = audios[0]
    
    if len(audio) > npu_static_length:
        print(f"Warning: Audio data {len(audio)} exceeds the target length {npu_static_length}. It will be trimmed to the target size.")
        trimmed_audio = audio[:npu_static_length]
        return [trimmed_audio]
    
    pad_length = npu_static_length - len(audio)
    padded_audio = np.concatenate([audio, np.zeros(pad_length, dtype=audio.dtype)])

    print(f"Warning: Audio data {len(audio)} smaller than the target length {npu_static_length}. It will be padding to the target size.")
    return [padded_audio]

def resize_image_for_npu(imgs, patch_size=14, npu_static_patch_length=2048):
    """
    Resize image for NPU processing by maintaining aspect ratio and padding.

    Args:
        imgs (list | None): Input image list.
        patch_size (int): Size of each patch.
        npu_static_patch_length (int): The static length for NPU processing.

    Returns:
        list | None: Resized and padded image in a list, or None if input is None.
    """
    if imgs is None:
        return None
    
    if not isinstance(imgs, list) or len(imgs) != 1:
        raise ValueError("Only one image supported for NPU. Please set device to GPU")
    
    img = imgs[0]
    
    orig_width, orig_height = img.size
    orig_ratio = orig_width / orig_height

    factor_pairs = [(2 ** i, 2048 // (2 ** i)) for i in range(12)]  # Generate factor pairs from 2^0 to 2^11

    min_diff = float('inf')
    best_a, best_b = 1, npu_static_patch_length
    for a, b in factor_pairs:
        current_ratio = b / a
        diff = abs(current_ratio - orig_ratio)
        if diff < min_diff:
            min_diff = diff
            best_a, best_b = a, b

    target_width = patch_size * best_b
    target_height = patch_size * best_a

    scale = min(target_width / orig_width, target_height / orig_height)
    new_width = int(round(orig_width * scale))
    new_height = int(round(orig_height * scale))

    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    new_img.paste(resized_img, (0, 0))

    print(f"Resize image: {orig_width}x{orig_height} -> {target_width}x{target_height} with patch number {npu_static_patch_length}")

    return [new_img]


def resize_images_for_gpu(images, patch_size=14, target_patch_size_each_img=2048):
    """
    Resize a list of images for GPU processing by maintaining aspect ratio.

    Args:
        images (list of PIL.Image.Image | None): List of input images.
        patch_size (int): Size of each patch.
        target_patch_size_each_img (int): Target patch size for each image.

    Returns:
        list of numpy.ndarray | None: List of processed images or None if input is None.
    """
    if images is None:
        return None
    
    processed_images = []
    for img in images:
        H, W = img.size
        
        max_product = target_patch_size_each_img * (patch_size ** 2)
        current_product = H * W
        
        scale = math.sqrt(max_product / current_product)
        scale = min(scale, 1.0)
        
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))
        
        resized_img = img.resize((new_W, new_H), Image.Resampling.LANCZOS)
        
        processed_images.append(np.array(resized_img))

        print(f"Resize image: {W}x{H} -> {new_W}x{new_H} with patch number {round(new_W*new_H/(patch_size ** 2))}")

    return processed_images

def resize_inputs(audios, images, videos, audio_len, img_patch_size, patch_length_per_img, device):
    if device == "NPU":
        audios = resize_audio_for_npu(audios, npu_static_length = audio_len)
        images = resize_image_for_npu(images, patch_size = img_patch_size, npu_static_patch_length = patch_length_per_img)

    if device != "NPU":
        images = resize_images_for_gpu(images, patch_size=img_patch_size, target_patch_size_each_img = patch_length_per_img)

    return audios, images, videos

def dump_inputs_info(inputs):
    print("=== Inputs Informations ===")
    # print(f"key values: {inputs.keys()}")
    # for key, value in inputs.items():
    #     print(f"\n{key} shape: {value.shape}")
    #     if hasattr(value, 'dtype'):
    #         print(f"{key} type: {value.dtype}")
    #     print(f"{key} value: {value}")

    if 'input_ids' in inputs:
        print(f"Total token length for Thinker LLM: {inputs['input_ids'].size(1)}")
    if 'pixel_values' in inputs:
        print(f" - Vision Embedding input length: {inputs['pixel_values'].size(0)}, output token length: {inputs['pixel_values'].size(0)/4}")
    if 'image_grid_thw' in inputs:
        print(f" - Vision Embedding include: {inputs['image_grid_thw'].size(0)} images")
    if 'feature_attention_mask' in inputs:
        print(f" - Audio Embedding input length: {inputs['feature_attention_mask'].sum(-1).item()}, output token length: {inputs['feature_attention_mask'].sum(-1).item()/4}")
