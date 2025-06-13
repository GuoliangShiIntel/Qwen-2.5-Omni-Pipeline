from qwen2_5_omni_helper import OVQwen2_5OmniModel
from transformers import Qwen2_5OmniProcessor
from transformers import TextStreamer

from qwen_omni_utils import process_mm_info

from pathlib import Path

import data_preprocess_helper as preprocess
import soundfile as sf

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
            # {"type": "audio", "audio": "inputs/Trailer.wav"},
        ],
    },
]

thinker_device = "GPU"
talker_device = "GPU"
token2wav_device = "CPU"

enable_talker = False

model_id = "Qwen/Qwen2.5-Omni-7B-INT4-SYM"
model_dir = Path(model_id.split("/")[-1])

processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

print("=== Chat Template ===")
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print(text)

audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
print("=== Resize Inputs ===")
audios, images, videos = preprocess.resize_inputs(audios, images, videos, audio_len=163839, img_patch_size=14, patch_length_per_img=2048, device=thinker_device)

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
preprocess.dump_inputs_info(inputs)

print("=== Compile And Load Models ===")
ov_model = OVQwen2_5OmniModel(model_dir, thinker_device=thinker_device, talker_device=talker_device, token2wav_device=token2wav_device, enable_talker=enable_talker)

if not enable_talker:
    text_ids = ov_model.generate(
        **inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), return_audio=enable_talker, thinker_max_new_tokens=1024
    )
else:
    text_ids, audio = ov_model.generate(
        **inputs, stream_config=TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True), return_audio=enable_talker, thinker_max_new_tokens=1024
    )

    sf.write("outputs/output.wav", audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
