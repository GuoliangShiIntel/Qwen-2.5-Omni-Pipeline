from qwen2_5_omni_helper import OVQwen2_5OmniModel
from transformers import Qwen2_5OmniProcessor
from transformers import TextStreamer

from notebook_utils import device_widget
from qwen_omni_utils import process_mm_info

from pathlib import Path

model_id = "Qwen/Qwen2.5-Omni-7B-INT4-SYM"
model_dir = Path(model_id.split("/")[-1])

thinker_device = device_widget(default="GPU", description="Thinker device")
talker_device = device_widget(default="GPU", description="Talker device")
token2wav_device = device_widget(default="CPU", description="Token2Wav device")

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
            # {"type": "text", "text": "Based on the content you have seen, what do you think I am doing?"},
            {"type": "text", "text": "These are the changes on the user's screen. What is the user's behavior?"},
            {"type": "image", "image": "inputs/user_behavior0_resize.png"},
            {"type": "image", "image": "inputs/user_behavior1_resize.png"},
            {"type": "image", "image": "inputs/user_behavior2_resize.png"},
        ],
    },
]

processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print("=== chat template ===")
print(text)

audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)

print("=== inputs informations ===")
print(f"key values: {inputs.keys()}")
for key, value in inputs.items():
    print(f"\n{key} shape: {value.shape}")
    # if hasattr(value, 'dtype'):
    #     print(f"{key} type: {value.dtype}")
    # print(f"{key} value: {value}")
print("=========================")

ov_model = OVQwen2_5OmniModel(model_dir, thinker_device=thinker_device.value, talker_device=talker_device.value, token2wav_device=token2wav_device.value, enable_talker=False)

text_ids = ov_model.generate(
    **inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), return_audio=False, thinker_max_new_tokens=1024
)
