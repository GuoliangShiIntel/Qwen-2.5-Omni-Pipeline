from qwen2_5_omni_helper import OVQwen2_5OmniModel
from transformers import Qwen2_5OmniProcessor
from transformers import TextStreamer

from qwen_omni_utils import process_mm_info

from pathlib import Path

import data_preprocess_helper as preprocess
import soundfile as sf
import keyboard
import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab, Image
import tkinter as tk
import torch
import time
import threading
import os
import queue

# Initialize models and processor
print("=== Compile And Load Models to Device ===")

thinker_device = "NPU"
talker_device = "GPU"
token2wav_device = "CPU"

enable_talker = False
use_audio_in_video = False
frame_selection_enabled = True  # Set to True to select only 2nd and 4th frames
selected_frame_indices = [1, 3]  # 0-based indices: [1, 3] means 2nd and 4th frames

model_id = "Qwen/Qwen2.5-Omni-3B-NF4"
model_dir = Path(model_id.split("/")[-1])

ov_model = OVQwen2_5OmniModel(model_dir, thinker_device=thinker_device, talker_device=talker_device, token2wav_device=token2wav_device, 
                              enable_talker=enable_talker, thinker_max_prompt_len=2048, thinker_min_response_len=256,
                              talker_max_prompt_len=1024, talker_min_response_len=256,
                              enable_cdpruner=True, cdpruner_num_visual_tokens=256, cdpruner_relevance_weight=0.5)

processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

def get_pytorch_device(ov_device):
    """Map OpenVINO device to PyTorch compatible device"""
    if ov_device == "NPU":
        return "cpu"
    elif ov_device == "GPU":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return ov_device.lower()

def select_video_frames(videos, frame_indices):
    """Select specific frames from videos based on indices"""
    if not videos or len(videos) == 0:
        return videos
    
    original_video = videos[0]
    
    # Handle different types of video data
    if hasattr(original_video, '__len__'):
        total_frames = len(original_video)
    else:
        return videos
    
    # Validate and filter indices
    valid_indices = [idx for idx in frame_indices if 0 <= idx < total_frames]
    
    if not valid_indices:
        return videos
    
    try:
        # Select frames - ensure we maintain the original structure
        if hasattr(original_video, 'shape'):
            # If it's a tensor or numpy array, use array indexing
            if isinstance(original_video, torch.Tensor):
                selected_frames = original_video[valid_indices]
            elif isinstance(original_video, np.ndarray):
                selected_frames = original_video[valid_indices]
            else:
                # Try array-like indexing
                selected_frames = np.array(original_video)[valid_indices]
        else:
            # If it's a list, select by indices
            selected_frames = [original_video[idx] for idx in valid_indices]
            
            # Try to convert to numpy array for consistency
            try:
                selected_frames = np.array(selected_frames)
            except Exception:
                pass
        
        print(f"Selected {len(valid_indices)} frames from {total_frames} total frames")
        
        return [selected_frames]
        
    except Exception as e:
        return videos

def select_area():
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.3)
    canvas = tk.Canvas(root, cursor="cross")
    canvas.pack(fill=tk.BOTH, expand=tk.YES)

    rect = None
    start_x = start_y = end_x = end_y = 0

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x = event.x
        start_y = event.y
        rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red')

    def on_move_press(event):
        nonlocal rect, end_x, end_y
        cur_x, cur_y = (event.x, event.y)
        width = abs(cur_x - start_x)
        height = abs(cur_y - start_y)

        if width > height:
            height = width // 2
        else:
            width = height // 2

        if cur_x < start_x:
            cur_x = start_x - width
        else:
            cur_x = start_x + width

        if cur_y < start_y:
            cur_y = start_y - height
        else:
            cur_y = start_y + height

        end_x, end_y = cur_x, cur_y
        canvas.coords(rect, start_x, start_y, cur_x, cur_y)

    def on_button_release(event):
        root.quit()

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    root.mainloop()
    root.destroy()

    return (start_x, start_y, end_x, end_y)

def capture_screen():
    x1, y1, x2, y2 = select_area()
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    img.save("inputs/screenshot.png")
    return "inputs/screenshot.png"

def record_screen():
    """Record screen with frame-based streaming video processing"""
    x1, y1, x2, y2 = select_area()
    
    # Setup video recording
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("inputs/screen_record.avi", fourcc, 20.0, (x2-x1, y2-y1))
    
    # Frame-based control settings
    fps = 20.0
    frames_per_segment = 20  # 20 frames = 1 second at 20fps
    target_seconds = 7  # Record for 7 seconds
    max_frames = target_seconds * frames_per_segment  # 7 * 20 = 140 frames total
    
    segment_frames = []
    processed_video_embeds = []
    
    # Thread-safe queues
    processing_queue = queue.Queue()
    results_queue = queue.Queue()
    
    total_frames_recorded = 0
    frame_count = 0
    recording_active = True
    start_time = time.time()
    
    def video_processor():
        """Background thread for processing video segments"""
        while True:
            try:
                item = processing_queue.get(timeout=1.0)
                if item is None:
                    break
                
                frames, width, height, segment_num = item
                
                segment_embed = process_video_segment(frames, width, height)
                if segment_embed is not None:
                    results_queue.put((segment_num, segment_embed))
                
                processing_queue.task_done()
                
            except queue.Empty:
                if not recording_active:
                    break
                continue
    
    # Start background processing thread
    processor_thread = threading.Thread(target=video_processor, daemon=True)
    processor_thread.start()
    
    print(f"=== Starting Frame-based Video Recording ===")
    print(f"Target: {target_seconds}s = {max_frames} frames")
    
    # Simple frame-based recording loop
    segment_counter = 0
    
    while total_frames_recorded < max_frames:
        # Check manual stop
        if keyboard.is_pressed("v"):
            print("Manual stop detected")
            break
        
        # Capture frame
        img = pyautogui.screenshot(region=(x1, y1, x2-x1, y2-y1))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        out.write(frame)
        
        # Store frame for processing
        segment_frames.append(frame)
        frame_count += 1
        total_frames_recorded += 1
        
        # Process complete segments (20 frames)
        if frame_count >= frames_per_segment:
            segment_counter += 1
            # Add segment to processing queue
            processing_queue.put((segment_frames.copy(), x2-x1, y2-y1, segment_counter))
            
            # Reset for next segment
            segment_frames = []
            frame_count = 0
    
    # Add any remaining frames to processing queue
    if segment_frames:
        segment_counter += 1
        processing_queue.put((segment_frames.copy(), x2-x1, y2-y1, segment_counter))
    
    out.release()
    recording_active = False
    
    final_time = time.time() - start_time
    actual_fps = total_frames_recorded / final_time if final_time > 0 else 0
    recording_end_time = time.time()
    
    print(f"Recording completed: {final_time:.2f}s, {total_frames_recorded} frames, {segment_counter} segments")
    print(f"Processing {segment_counter} segments...")
    
    # Wait for all processing to complete
    processing_queue.join()
    
    # Stop the processor thread
    processing_queue.put(None)
    processor_thread.join()
    
    processing_end_time = time.time()
    processing_gap = processing_end_time - recording_end_time
    
    # Collect all results
    results = {}
    while not results_queue.empty():
        segment_num, embed = results_queue.get()
        results[segment_num] = embed
    
    # Sort results by segment number and create final embeddings
    if results:
        sorted_segments = sorted(results.keys())
        processed_video_embeds = [results[i] for i in sorted_segments]
        final_video_embeds = torch.cat(processed_video_embeds, dim=0)
        print(f"Processing completed in {processing_gap:.2f}s")
        print(f"Final video embeds shape: {final_video_embeds.shape}")
        return "inputs/screen_record.avi", final_video_embeds
    else:
        print(f"Processing completed in {processing_gap:.2f}s (no results)")
        return "inputs/screen_record.avi", None

def process_video_segment(frames, width, height):
    """Process a 1-second video segment to get video embeddings"""
    try:
        # Create a temporary video file for this segment
        temp_video_path = f"inputs/temp_segment_{int(time.time()*1000)}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        temp_out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (width, height))
        
        for frame in frames:
            temp_out.write(frame)
        temp_out.release()
        
        # Create a temporary conversation for this segment
        temp_conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Processing video segment."}]
            },
            {
                "role": "user", 
                "content": [{"type": "video", "video": temp_video_path}]
            }
        ]
        
        # Process the segment
        audios, images, videos = process_mm_info(temp_conversation, use_audio_in_video=use_audio_in_video)
        
        # Filter videos to keep only selected frames
        if frame_selection_enabled and videos is not None:
            videos = select_video_frames(videos, selected_frame_indices)
        
        audios, images, videos = preprocess.resize_inputs(audios, images, videos, audio_len=163839, img_patch_size=14, patch_length_per_img=2048, device=get_pytorch_device(thinker_device))
        
        # Get processor inputs
        inputs = processor(text="", audio=audios, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
        
        # Extract video processing data
        pixel_values_videos = inputs.get('pixel_values_videos')
        video_grid_thw = inputs.get('video_grid_thw')
        
        if pixel_values_videos is not None and video_grid_thw is not None:
            # Use the vision processor to get video embeddings
            pytorch_device = get_pytorch_device(thinker_device)
            pixel_values_videos = pixel_values_videos.to(pytorch_device)
            video_grid_thw = video_grid_thw.to(pytorch_device)
            
            # Access the vision processor from the model
            vision_processor = ov_model.thinker.vision_processor
            video_embed = vision_processor.process_visual_features(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_video_path)
            except:
                pass
                
            return video_embed
        
        # Clean up temporary file if processing failed
        try:
            os.remove(temp_video_path)
        except:
            pass
            
        return None
        
    except Exception as e:
        # Clean up temporary file in case of error
        try:
            if 'temp_video_path' in locals():
                os.remove(temp_video_path)
        except:
            pass
        return None

def process_conversation(conversation, preprocessed_video_embeds=None):
    print("=== Chat Template ===")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print(text)

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    audios, images, videos = preprocess.resize_inputs(audios, images, videos, audio_len=163839, img_patch_size=14, patch_length_per_img=2048, device=get_pytorch_device(thinker_device))

    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)

    preprocess.dump_inputs_info(inputs)

    print("=== Infer and get Result ===")

    # Add preprocessed video embeddings if available
    generate_kwargs = {
        "stream_config": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True),
        "use_audio_in_video": use_audio_in_video,
        "return_audio": enable_talker,
        "thinker_max_new_tokens": 1024
    }
    
    if preprocessed_video_embeds is not None:
        generate_kwargs["preprocessed_video_embeds"] = preprocessed_video_embeds
        print(f"Using preprocessed video embeddings with shape: {preprocessed_video_embeds.shape}")

    if not enable_talker:
        text_ids = ov_model.generate(**inputs, **generate_kwargs)
    else:
        text_ids, audio = ov_model.generate(**inputs, **generate_kwargs)
        
        # Ensure the outputs directory exists
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sf.write("outputs/output.wav", audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)

    conversation[1]["content"] = [item for item in conversation[1]["content"] if item["type"] == "text"]

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }
        ],
    },
    {
        "role": "user",
        # "content": [
        #     {"type": "text", "text": "请你描述我在干什么？1. 如果发现我遇到了什么问题，请给出一些具体的解决方案？"
        #     "2. 如果我在看一些文章或者浏览网页，请对里面的内容进行翻译/归纳/总结，不要太简略，又能让我更快获取重要信息。"
        #     "3. 如果我在购物，请给出我想购买产品的对比信息和一些重要参数，并给出你的推荐。"},
        # ],
        # "content": [
        #     {"type": "text", "text": "请详细描述这段视频的内容。这段视频展示了一个论文的基本介绍页面，包括以下部分："
        #                             "1. 摘要 ：请详细描述论文的摘要内容，包括研究的背景、目的、方法和主要发现。"
        #                             "2. 框架 ：请详细描述论文的框架图，包括各个模块的功能和相互关系。"
        #                             "3. 效果图 ：请详细描述论文中的效果图，包括图中的数据、图表类型和所展示的结果。"
        #                             "4. 性能比较 ：请详细描述论文中的性能比较部分，包括比较的指标、方法和结果。"
        #                             "请确保描述中包含所有细节，并且每个部分都清晰明了。"},
        # ],
        "content": [
            {"type": "text", "text": "请详细描述你看到的所有内容，包括且不限于："
             "1. 打印出所有的文字；2.详细解释图片中的架构和细节；3.详细解释图标中的数据，并给出具体值；4.做一个综述总结；每一条为一个段落"},
        ],
    },
]

print("=== Ready to Capture and Record ===")
print("Press [SPACE] for screenshot, [V] for streaming video recording, [ESC] to exit")

if frame_selection_enabled:
    frame_numbers = [idx + 1 for idx in selected_frame_indices]
    print(f"Frame selection enabled: Only frames {frame_numbers} will be processed from each segment")
else:
    print("Frame selection disabled: All frames will be processed")

while True:
    if keyboard.is_pressed("space"):
        image_path = capture_screen()
        conversation[1]["content"].append({"type": "image", "image": image_path})
        process_conversation(conversation)
    elif keyboard.is_pressed("v"):
        # Use streaming video processing
        video_path, preprocessed_video_embeds = record_screen()
        conversation[1]["content"].append({"type": "video", "video": video_path})
        
        # Process conversation with preprocessed video embeddings
        process_conversation(conversation, preprocessed_video_embeds)
    elif keyboard.is_pressed("esc"):
        print("=== Exiting ===")
        break
