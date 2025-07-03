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

# Initialize models and processor
print("=== Compile And Load Models to Device ===")

thinker_device = "NPU"
talker_device = "GPU"
token2wav_device = "CPU"

enable_talker = False
use_audio_in_video = False

model_id = "Qwen/Qwen2.5-Omni-7B-NF4"
model_dir = Path(model_id.split("/")[-1])

ov_model = OVQwen2_5OmniModel(model_dir, thinker_device=thinker_device, talker_device=talker_device, token2wav_device=token2wav_device, enable_talker=enable_talker)
processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

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
    x1, y1, x2, y2 = select_area()
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("inputs/screen_record.avi", fourcc, 20.0, (x2-x1, y2-y1))
    
    start_time = cv2.getTickCount()
    max_duration = 7.8 * cv2.getTickFrequency()
    
    while True:
        img = pyautogui.screenshot(region=(x1, y1, x2-x1, y2-y1))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        out.write(frame)
        
        if keyboard.is_pressed("v"):
            break
        
        if (cv2.getTickCount() - start_time) > max_duration:
            break
    
    out.release()
    return "inputs/screen_record.avi"

def process_conversation(conversation):
    print("=== Chat Template ===")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print(text)

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    audios, images, videos = preprocess.resize_inputs(audios, images, videos, audio_len=163839, img_patch_size=14, patch_length_per_img=2048, device=thinker_device)

    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)

    # preprocess.dump_inputs_info(inputs)

    print("=== Infer and get Result ===")

    if not enable_talker:
        text_ids = ov_model.generate(
            **inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), use_audio_in_video=use_audio_in_video, return_audio=enable_talker, thinker_max_new_tokens=1024
        )
    else:
        text_ids, audio = ov_model.generate(
            ** inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), use_audio_in_video=use_audio_in_video, return_audio=enable_talker, thinker_max_new_tokens=1024
        )

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
        "content": [
            {"type": "text", "text": "请你描述我在干什么？1. 如果发现我遇到了什么问题，请给出一些具体的解决方案？"
            "2. 如果我在看一些文章或者浏览网页，请对里面的内容进行翻译/归纳/总结，不要太简略，又能让我更快获取重要信息。"
            "2. 如果我在购物，请给出我想购买产品的对比信息和一些重要参数，并给出你的推荐。"},
        ],
    },
]

print("=== Ready to Capture and Record ===")

while True:
    if keyboard.is_pressed("space"):
        image_path = capture_screen()
        conversation[1]["content"].append({"type": "image", "image": image_path})
        process_conversation(conversation)
    elif keyboard.is_pressed("v"):
        video_path = record_screen()
        conversation[1]["content"].append({"type": "video", "video": video_path})
        process_conversation(conversation)
    elif keyboard.is_pressed("esc"):
        print("=== Exiting ===")
        break
