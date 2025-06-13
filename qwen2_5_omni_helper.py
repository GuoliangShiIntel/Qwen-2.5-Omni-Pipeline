from pathlib import Path
import openvino as ov
import time

from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
import numpy as np
import operator
import torch
from transformers import AutoConfig
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniTalkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    ALL_ATTENTION_FUNCTIONS
)
from pathlib import Path
from itertools import accumulate
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from typing import Optional, Union, Any

import compile_ov_model_helper as ov_compiler

core = ov.Core()

THINKER_EMBEDDING_NAME = "openvino_thinker_embedding_model.xml"
THINKER_AUDIO_NAME = "openvino_thinker_audio_model.xml"
THINKER_AUDIO_STATE_NAME = "openvino_thinker_audio_state_model.xml"
THINKER_AUDIO_EMBED_NAME = "openvino_thinker_audio_embed_model.xml"
THINKER_PATCHER_NAME = "openvino_thinker_patcher_model.xml"
THINKER_MERGER_NAME = "openvino_thinker_merger_model.xml"
THINKER_LANGUAGE_NAME = "openvino_thinker_language_model.xml"

TALKER_LANGUAGE_NAME = "openvino_talker_language_model.xml"
TALKER_EMBEDDING_NAME = "openvino_talker_embedding_model.xml"

TOKEN2WAV_DIT_NAME = "openvino_token2wav_dit_model.xml"
TOKEN2WAV_BIGVGAN_NAME = "openvino_token2wav_bigvgan_model.xml"


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[int],
    grid_hs: list[int],
    grid_ws: list[int],
):
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
    # + 1 ) # 12.09 by malinhan
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def get_chunked_index(llm_pos_ids, t_ntoken_per_chunk, st_idx):
    def _iter():
        i, start_idx = 0, 0  # skip bos token
        current_chunk = 1
        while i < llm_pos_ids.shape[1]:  # skip eos token
            if llm_pos_ids[0][i] - st_idx >= current_chunk * t_ntoken_per_chunk:
                yield (start_idx, i)
                start_idx = i
                current_chunk += 1
            i += 1
        yield (start_idx, llm_pos_ids.shape[1])

    return list(_iter())


def get_rope_index(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(config, "vision_config"):
        spatial_merge_size = config.vision_config.spatial_merge_size
    else:
        spatial_merge_size = config.spatial_merge_size
    image_token_id = config.image_token_index
    video_token_id = config.video_token_index
    audio_token_id = config.audio_token_index
    vision_start_token_id = config.vision_start_token_id
    audio_start_token_id = config.audio_start_token_id
    position_id_per_seconds = config.position_id_per_seconds
    seconds_per_chunk = config.seconds_per_chunk

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums, audio_nums = 0, 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == audio_start_token_id).sum() if use_audio_in_video else (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                min_ed = min(ed_image, ed_video, ed_audio)
                if min_ed == ed_audio:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_idx += 1
                    remain_audios -= 1

                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws)
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video and not use_audio_in_video:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws)
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + video_len + eos_len
                    video_idx += 1
                    remain_videos -= 1

                elif min_ed == ed_video and use_audio_in_video:
                    text_len = min_ed - st - 2
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]

                    t_index = (torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds).long()
                    video_llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws)

                    t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                    video_chunk_indexes = get_chunked_index(video_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                    audio_chunk_indexes = get_chunked_index(audio_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                    sub_len = 0
                    for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                        video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                        audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                        if video_chunk_index is not None:
                            sub_len += video_chunk_index[1] - video_chunk_index[0]

                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]])
                        if audio_chunk_index is not None:
                            sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]])
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas


class OVQwen2_5OmniThinkerForConditionalGeneration(GenerationMixin):
    def __init__(self, model_dir, device, config):
        self.infer_device = device
        self.model = core.read_model(model_dir / THINKER_LANGUAGE_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}

        if device == "NPU":
            # Audio Embedding
            audio_embed_blob_cache_path = model_dir / ".blob_cache" / "thinker_audio_embedding.blob"
            self.audio_embed = ov_compiler.npu_model_import_or_compile(audio_embed_blob_cache_path, model_dir / THINKER_AUDIO_EMBED_NAME, ov_compiler.convert_thinker_audio_embedding_to_static_shape, device, 'thinker_audio_embedding')
            audio_blob_cache_path = model_dir / ".blob_cache" / "thinker_audio.blob"
            self.audio = ov_compiler.npu_model_import_or_compile(audio_blob_cache_path, model_dir / THINKER_AUDIO_NAME, ov_compiler.convert_thinker_audio_to_static_shape, device, 'thinker_audio')
            audio_state_blob_cache_path = model_dir / ".blob_cache" / "thinker_audio_state.blob"
            self.audio_state = ov_compiler.npu_model_import_or_compile(audio_state_blob_cache_path, model_dir / THINKER_AUDIO_STATE_NAME, ov_compiler.convert_thinker_audio_state_to_static_shape, device, 'thinker_audio_state')
            # Vision Embedding
            vision_merger_blob_cache_path = model_dir / ".blob_cache" / "thinker_vision_merger.blob"
            self.visual_merger = ov_compiler.npu_model_import_or_compile(vision_merger_blob_cache_path, model_dir / THINKER_MERGER_NAME, ov_compiler.convert_thinker_vision_merger_to_static_shape, device, 'thinker_vision_merger')
            # LLM
            llm_blob_cache_path = model_dir / ".blob_cache" / "thinker_language_npuw.blob"
            weights_bin = model_dir / "openvino_thinker_language_model.bin"
            llm = ov_compiler.npu_llm_model_import_or_compile(llm_blob_cache_path, model_dir / THINKER_LANGUAGE_NAME, weights_bin, device, 'thinker_language')
            self.request = llm.create_infer_request()
        else:
            # Audio Embedding
            audio_embed_blob_cache_path = model_dir / ".blob_cache" / f"thinker_audio_embedding_{device}.blob"
            self.audio_embed = ov_compiler.cpu_gpu_model_import_or_compile(audio_embed_blob_cache_path, model_dir / THINKER_AUDIO_EMBED_NAME, device, 'thinker_audio_embedding')
            audio_blob_cache_path = model_dir / ".blob_cache" / f"thinker_audio_{device}.blob"
            self.audio = ov_compiler.cpu_gpu_model_import_or_compile(audio_blob_cache_path, model_dir / THINKER_AUDIO_NAME, device, 'thinker_audio')
            audio_state_blob_cache_path = model_dir / ".blob_cache" / f"thinker_audio_state_{device}.blob"
            self.audio_state = ov_compiler.cpu_gpu_model_import_or_compile(audio_state_blob_cache_path, model_dir / THINKER_AUDIO_STATE_NAME, device, 'thinker_audio_state')
            # Vision Embedding
            vision_merger_blob_cache_path = model_dir / ".blob_cache" / f"thinker_vision_merger_{device}.blob"
            self.visual_merger = ov_compiler.cpu_gpu_model_import_or_compile(vision_merger_blob_cache_path, model_dir / THINKER_MERGER_NAME, device, 'thinker_vision_merger')
            # LLM
            llm_blob_cache_path = model_dir / ".blob_cache" / f"thinker_language_{device}.blob"
            llm = ov_compiler.cpu_gpu_model_import_or_compile(llm_blob_cache_path, model_dir / THINKER_LANGUAGE_NAME, device, 'thinker_language', False)
            self.request = llm.create_infer_request()

        # Two dymanic small graph, always run on CPU
        text_embedding_blob_cache_path = model_dir / ".blob_cache" / f"thinker_embedding_CPU.blob"
        self.embed_tokens = ov_compiler.cpu_gpu_model_import_or_compile(text_embedding_blob_cache_path, model_dir / THINKER_EMBEDDING_NAME, 'CPU', 'thinker_embedding')
        vision_blob_cache_path = model_dir / ".blob_cache" / f"thinker_vision_CPU.blob"
        self.visual_patcher = ov_compiler.cpu_gpu_model_import_or_compile(vision_blob_cache_path, model_dir / THINKER_PATCHER_NAME, 'CPU', 'thinker_vision')

        self.main_input_name = "input_ids"
        self.config = config.thinker_config
        self.n_window = self.config.audio_config.n_window
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self._past_length = None
        self.next_beam_idx = None
        self.spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.fullatt_block_indexes = self.config.vision_config.fullatt_block_indexes
        self.window_size = self.config.vision_config.window_size
        self.spatial_merge_unit = self.config.vision_config.spatial_merge_size * self.config.vision_config.spatial_merge_size
        self._skip_keys_device_placement = "past_key_values"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = True
        self._supports_cache_class = True
        self._supports_static_cache = True

        self.llm_times = []
        class Qwen2_5_VisionRotaryEmbedding(torch.nn.Module):
            def __init__(self, dim: int, theta: float = 10000.0) -> None:
                super().__init__()
                inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            def forward(self, seqlen: int) -> torch.Tensor:
                seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(seq, self.inv_freq)
                return freqs

        head_dim = self.config.vision_config.hidden_size // self.config.vision_config.num_heads
        self._rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        if past_key_values != ((),):
            past_key_values = None
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self._rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def visual(self, pixel_values, grid_thw, **kwargs):
        visual_patcher_start_time = time.perf_counter()
        hidden_states = self.visual_patcher(pixel_values)[0] # [Thinker][Vision][Modle ID 0]
        print(f"[Thinker][Vision_ID_0][CPU] visual patcher infer time: {(time.perf_counter() - visual_patcher_start_time)*1000} ms")
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        attention_mask = torch.zeros((1, hidden_states.shape[0], hidden_states.shape[0]), dtype=torch.bool)
        causal_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        causal_mask.masked_fill_(torch.logical_not(attention_mask), float("-inf"))

        window_attention_mask = torch.zeros((1, hidden_states.shape[0], hidden_states.shape[0]), dtype=torch.bool)
        window_causal_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
        for i in range(1, len(cu_window_seqlens)):
            window_attention_mask[..., cu_window_seqlens[i - 1] : cu_window_seqlens[i], cu_window_seqlens[i - 1] : cu_window_seqlens[i]] = True

        window_causal_mask.masked_fill_(torch.logical_not(window_attention_mask), float("-inf"))

        visual_merger_start_time = time.perf_counter()
        res = self.visual_merger([hidden_states, causal_mask, window_causal_mask, window_index, rotary_pos_emb])[0] # [Thinker][Vision][Modle ID 1]
        print(f"[Thinker][Vision_ID_1][{self.infer_device}] visual merger infer time: {(time.perf_counter() - visual_merger_start_time)*1000} ms")
        return torch.from_numpy(res)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = get_rope_index(
                    self.config,
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = torch.from_numpy(self.embed_tokens(input_ids)[0]) # [Thinker][Text][Modle ID 0]
        if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
            if input_features is not None:
                audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )
                feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

                chunk_lengths = torch.tensor(
                    [self.n_window * 2] * chunk_num.sum(),
                    dtype=torch.long,
                    device=feature_lens.device,
                )
                tail_chunk_index = list(accumulate(chunk_num.tolist(), func=operator.add, initial=-1))[1:]
                chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
                chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

                chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)

                padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
                    chunk_list, chunk_lengths, padding_value=0, padding_side="right"
                )
                audio_embed_start_time = time.perf_counter()
                padded_embed = torch.from_numpy(self.audio_embed([padded_feature, padded_mask])[0]) # [Thinker][Audio][Modle ID 0]
                print(f"[Thinker][Audio_ID_0][{self.infer_device}] audio embed infer time: {(time.perf_counter() - audio_embed_start_time)*1000} ms")
                hidden_states = padded_embed[padded_mask_after_cnn.bool()]
                
                audio_start_time = time.perf_counter()
                hidden_states = torch.from_numpy(self.audio([hidden_states, padded_mask_after_cnn])[0]) # [Thinker][Audio][Modle ID 1]
                print(f"[Thinker][Audio_ID_1][{self.infer_device}] audio infer time: {(time.perf_counter() - audio_start_time)*1000} ms")
                hidden_states_list = hidden_states.split(audio_feat_lengths.tolist(), dim=0)
                token_audio_list = []
                for each_audio_states in hidden_states_list:
                    audio_state_start_time = time.perf_counter()
                    each_audio_states = torch.from_numpy(self.audio_state([each_audio_states])[0]) # [Thinker][Audio][Modle ID 2]
                    print(f"[Thinker][Audio_ID_2][{self.infer_device}] audio_state infer time: {(time.perf_counter() - audio_state_start_time)*1000} ms")
                    token_audio_list.append(each_audio_states)
                audio_features = torch.cat(token_audio_list, dim=0)

                if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                    raise ValueError("length of audio_features should match audio_output_lengths")
                audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if pixel_values is not None:
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(inputs_embeds.shape[0], dtype=int)
            self._past_length = 0
        inputs = {}
        inputs["inputs_embeds"] = inputs_embeds
        inputs["attention_mask"] = attention_mask
        inputs["position_ids"] = position_ids
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(inputs_embeds.shape[0], dtype=int)
        llm_start_time = time.perf_counter()
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        self.llm_times.append(time.perf_counter() - llm_start_time)
        logits = self.request.get_tensor("logits").data
        hidden_states = self.request.get_tensor("hidden_states").data
        # Specific slice for NPU
        if hidden_states.shape[1] != 1:
            hidden_states = hidden_states[:, -1 * inputs_embeds.shape[1]:, :]

        logits = torch.from_numpy(logits).to(self.device)
        hidden_states = torch.from_numpy(hidden_states).to(self.device)
        past_key_values = ((),)
        embeds_to_talker = inputs_embeds.clone()
        hidden_states_output = hidden_states.clone()
        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            logits=logits, past_key_values=past_key_values, rope_deltas=rope_deltas, hidden_states=(embeds_to_talker, hidden_states_output)
        )

    def _reorder_cache(self, past_key_values: tuple[tuple[torch.Tensor]], beam_idx: torch.Tensor) -> tuple[tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    # Copied from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1602

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs


class OVQwen2_5OmniTalkerForConditionalGeneration(GenerationMixin):
    def __init__(self, model_dir, device, config):
        self.model = core.read_model(model_dir / TALKER_LANGUAGE_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}

        if device == "NPU":
            llm_blob_cache_path = model_dir / ".blob_cache" / "talker_language_npuw.blob"
            weights_bin = model_dir / "openvino_talker_language_model.bin"
            llm = ov_compiler.npu_llm_model_import_or_compile(llm_blob_cache_path, model_dir / TALKER_LANGUAGE_NAME, weights_bin, device, 'talker_language')
            self.request = llm.create_infer_request()
        else:
            llm_blob_cache_path = model_dir / ".blob_cache" / f"talker_language_{device}.blob"
            llm = ov_compiler.cpu_gpu_model_import_or_compile(llm_blob_cache_path, model_dir / TALKER_LANGUAGE_NAME, device, 'talker_language', False)
            self.request = llm.create_infer_request()

        text_embedding_blob_cache_path = model_dir / ".blob_cache" / f"talker_embedding_CPU.blob"
        self.embed_tokens = ov_compiler.cpu_gpu_model_import_or_compile(text_embedding_blob_cache_path, model_dir / TALKER_EMBEDDING_NAME, 'CPU', 'talker_embedding')

        self.config = config.talker_config
        self.generation_config = GenerationConfig.from_model_config(self.config)

        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self._past_length = None
        self.next_beam_idx = None
        self._skip_keys_device_placement = "past_key_values"
        self._supports_flash_attn_2 = True
        self._supports_sdpa = True
        self._supports_cache_class = True
        self._supports_static_cache = True
        self.codebook_size = self.config.vocab_size
        self.codec_bos_token = self.config.tts_codec_start_token_id
        self.codec_eos_token = self.config.tts_codec_end_token_id
        self.codec_pad_token = self.config.tts_codec_pad_token_id
        self.codec_mask_token = self.config.tts_codec_mask_token_id

        self.text_bos_token = self.config.tts_text_start_token_id
        self.text_eos_token = self.config.tts_text_end_token_id
        self.text_pad_token = self.config.tts_text_pad_token_id

        self.spatial_merge_size = self.config.spatial_merge_size

        self.llm_times = []

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        input_text_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids=input_ids,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        input_text_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_audio_in_video: Optional[bool] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = get_rope_index(
                    self.config,
                    input_text_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                inputs_embeds[:, -1, :] += torch.from_numpy(
                    self.embed_tokens(torch.tensor([[self.codec_bos_token]], dtype=torch.long, device=inputs_embeds.device))[0][0]
                )
                inputs_embeds[:, -2, :] += torch.from_numpy(
                    self.embed_tokens(torch.tensor([[self.codec_pad_token]], dtype=torch.long, device=inputs_embeds.device))[0][0]
                )

            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. 推理第 2 个以及之后的 token
            codec_embeds = torch.from_numpy(self.embed_tokens(input_ids)[0])
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]
        if past_key_values is None:
            self.request.reset_state()
            self.next_beam_idx = np.arange(inputs_embeds.shape[0], dtype=int)
            self._past_length = 0
        inputs = {}
        inputs["inputs_embeds"] = inputs_embeds
        inputs["attention_mask"] = attention_mask
        inputs["position_ids"] = position_ids
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(inputs_embeds.shape[0], dtype=int)
        llm_start_time = time.perf_counter()
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        self.llm_times.append(time.perf_counter() - llm_start_time)
        logits = self.request.get_tensor("logits").data
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)

        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            rope_deltas=rope_deltas,
            thinker_reply_part=thinker_reply_part,
        )

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        # Talker needs to calculate cache_position with input_ids, so pop inputs_embeds temporarily
        inputs_embeds = model_kwargs.pop("inputs_embeds")
        model_kwargs = super()._get_initial_cache_position(input_ids, model_kwargs)
        model_kwargs["inputs_embeds"] = inputs_embeds
        return model_kwargs

    # prepare inputs for talker lm generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        if past_key_values != ((),):
            past_key_values = None
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            use_cache=use_cache,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
        model_inputs["position_ids"] = None

        return model_inputs

    def _reorder_cache(self, past_key_values: tuple[tuple[torch.Tensor]], beam_idx: torch.Tensor) -> tuple[tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        # update attention_mask
        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        if getattr(outputs, "thinker_reply_part", None) is not None:
            model_kwargs["thinker_reply_part"] = outputs.thinker_reply_part

        return model_kwargs


class RungeKutta4ODESolver:
    def __init__(self, function, initial_value):
        self.function = function
        self.initial_value = initial_value

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_step(self, function, time_start, time_step, time_end, value_start, function_value_start=None):
        k1 = function_value_start if function_value_start is not None else function(time_start, value_start)
        k2 = function(time_start + time_step * self._one_third, value_start + time_step * k1 * self._one_third)
        k3 = function(time_start + time_step * self._two_thirds, value_start + time_step * (k2 - k1 * self._one_third))
        k4 = function(time_end, value_start + time_step * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * time_step / 8

    def _compute_step(self, function, time_start, time_step, time_end, value_start):
        function_value_start = function(time_start, value_start)
        return self._rk4_step(function, time_start, time_step, time_end, value_start, function_value_start=function_value_start), function_value_start

    def _linear_interpolation(self, time_start, time_end, value_start, value_end, time_point):
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points):
        solution = torch.empty(
            len(time_points),
            *self.initial_value.shape,
            dtype=self.initial_value.dtype,
            device=self.initial_value.device,
        )
        solution[0] = self.initial_value

        current_index = 1
        current_value = self.initial_value
        for time_start, time_end in zip(time_points[:-1], time_points[1:]):
            time_step = time_end - time_start
            delta_value, _ = self._compute_step(self.function, time_start, time_step, time_end, current_value)
            next_value = current_value + delta_value

            while current_index < len(time_points) and time_end >= time_points[current_index]:
                solution[current_index] = self._linear_interpolation(time_start, time_end, current_value, next_value, time_points[current_index])
                current_index += 1

            current_value = next_value

        return solution


class OVQwen2_5OmniModel(GenerationMixin):
    def __init__(self, model_dir, thinker_device, talker_device, token2wav_device, enable_talker):
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.thinker_infer_device = thinker_device
        self.talker_infer_device = talker_device
        self.token2wav_infer_device = token2wav_device

        self.has_talker = enable_talker
        model_path = Path(model_dir)
        self.thinker = OVQwen2_5OmniThinkerForConditionalGeneration(model_path / "thinker", thinker_device, self.config)
        self.speaker_map = {}
        if self.has_talker:
            self.enable_talker(model_path, talker_device, token2wav_device)
        spk_path = model_path / "spk_dict.pt"

        self.load_speakers(spk_path)

    def enable_talker(self, model_path, talker_device, token2wav_device=None):
        if token2wav_device is None:
            token2wav_device = talker_device
        self.talker = OVQwen2_5OmniTalkerForConditionalGeneration(model_path / "talker", talker_device, self.config)

        if token2wav_device == 'NPU':
            token2wav_dit_blob_cache_path = model_path / ".blob_cache" / "token2wav_dit.blob"
            self.token2wav_dit = ov_compiler.npu_model_import_or_compile(token2wav_dit_blob_cache_path, model_path / TOKEN2WAV_DIT_NAME, ov_compiler.convert_token2wav_dit_to_static_shape, token2wav_device, 'token2wav_dit')
            token2wav_bigvgan_blob_cache_path = model_path / ".blob_cache" / "token2wav_bigvgan.blob"
            self.token2wav_bigvgan = ov_compiler.npu_model_import_or_compile(token2wav_bigvgan_blob_cache_path, model_path / TOKEN2WAV_BIGVGAN_NAME, ov_compiler.convert_token2wav_bigvgan_to_static_shape, token2wav_device, 'token2wav_bigvgan')
        else:
            token2wav_dit_blob_cache_path = model_path / ".blob_cache" / f"token2wav_dit_{token2wav_device}.blob"
            self.token2wav_dit = ov_compiler.cpu_gpu_model_import_or_compile(token2wav_dit_blob_cache_path, model_path / TOKEN2WAV_DIT_NAME, token2wav_device, 'token2wav_dit')
            token2wav_bigvgan_blob_cache_path = model_path / ".blob_cache" / f"token2wav_bigvgan_{token2wav_device}.blob"
            self.token2wav_bigvgan = ov_compiler.cpu_gpu_model_import_or_compile(token2wav_bigvgan_blob_cache_path, model_path / TOKEN2WAV_BIGVGAN_NAME, token2wav_device, 'token2wav_bigvgan')

        self.has_talker = True

    def load_speakers(self, path):
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav"):
            del self.token2wav
        self.has_talker = False

    @classmethod
    def can_generate(cls) -> bool:
        return True

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.tensor] = None,
        speaker: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        stream_config=None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            spk (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        if speaker not in self.speaker_map:
            raise ValueError(f"{speaker} is not availible, availible speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError("Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker.")
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value
        speaker_params = self.speaker_map[speaker]

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True
        if stream_config is not None:
            thinker_kwargs["streamer"] = stream_config
        print("[===start thinker===]")
        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)
        print(f"[Thinker][LLM] Input Shape: {input_ids.shape}")

        llm_thinker_times = self.thinker.llm_times
        print(f"[Thinker][LLM_Prefill][{self.thinker_infer_device}] Infer time: {llm_thinker_times[0]*1000} ms")
        remaining_list = llm_thinker_times[1:]
        average = sum(remaining_list) / len(remaining_list)
        print(f"[Thinker][LLM_KV_CACHE][{self.thinker_infer_device}] Infer: {1 / average} token/s")

        if not generate_audio:
            return thinker_result

        # 2. Generate speech tokens from talker module
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(self.talker.device)
        thinker_token_embeds = [x[0].to(self.talker.device) for x in thinker_result.hidden_states]
        thinker_hidden_states = [x[1].to(self.talker.device) for x in thinker_result.hidden_states]
        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(self.talker.device),
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.talker.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token, device=self.talker.device),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.talker.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.talker.device),
            ],
            dim=1,
        )

        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device)
        talker_text_bos_embed = torch.from_numpy(self.thinker.embed_tokens(talker_text_bos_token)[0]).to(self.talker.device)

        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_embedding = torch.from_numpy(
            self.thinker.embed_tokens(torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.thinker.device))[0]
        ).to(self.talker.device)

        pad_embedding = torch.from_numpy(
            self.thinker.embed_tokens(torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.thinker.device))[0]
        ).to(self.talker.device)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )

        talker_attention_mask = None
        if "attention_mask" in kwargs:
            talker_attention_mask = torch.cat([kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1).to(self.talker.device)
        print("[===start talker===]")
        print(f"[Talker][LLM] Input Shape: {talker_input_ids.shape}")
        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{k: (v.to(self.talker.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
        )
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]
        print(f"[Talker][LLM] Generate Shape: {talker_generate_codes.shape}")

        llm_talker_times = self.talker.llm_times
        print(f"[Talker][LLM_Prefill][{self.talker_infer_device}] Infer time: {llm_talker_times[0]*1000} ms")
        remaining_list = llm_talker_times[1:]
        average = sum(remaining_list) / len(remaining_list)
        print(f"[Talker][LLM_KV_CACHE][{self.talker_infer_device}] Infer: {1 / average} token/s")

        print("[===start token2wav===]")
        # 3. Generate wavs from code
        reference_mel_spectrogram = speaker_params["ref_mel"].to(torch.device("cpu")).float()
        conditioning_vector = speaker_params["cond"].to(torch.device("cpu")).float()
        noise_initialization = torch.randn([1, 30000, self.config.token2wav_config.dit_config.mel_dim], dtype=reference_mel_spectrogram.dtype)
        maximum_duration = talker_generate_codes.shape[1] * self.config.token2wav_config.dit_config.repeats
        initial_state = noise_initialization[:, :maximum_duration].to(talker_generate_codes.device)
        batch_size = reference_mel_spectrogram.shape[0]
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)
        if batch_size != 1:
            raise ValueError("Only batch size = 1 is currently supported")
        guidance_scale = 0.5
        sway_coefficient = -1.0

        def ode_function(time_step, hidden_states):
            token2wav_dit_start_time = time.perf_counter()
            model_output = torch.from_numpy(
                self.token2wav_dit([hidden_states, reference_mel_spectrogram, conditioning_vector, talker_generate_codes, time_step])[0]
            )
            print(f"[Token2wav][Model_ID_0][{self.token2wav_infer_device}] token2wav_dit infer time: {(time.perf_counter() - token2wav_dit_start_time)*1000} ms")
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)

            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        initial_time = 0
        time_embedding = torch.linspace(initial_time, 1, 10, device=talker_generate_codes.device, dtype=conditioning_vector.dtype)

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        print(f"[Token2wav][Model_ID_0][hidden_states] Shape {initial_state.shape}")
        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        print(f"[Token2wav][Model_ID_1][mel_spectrogram] Shape {generated_mel_spectrogram.shape}")
        token2wav_bigvgan_start_time = time.perf_counter()
        waveform = torch.from_numpy(self.token2wav_bigvgan([generated_mel_spectrogram])[0])
        print(f"[Token2wav][Model_ID_1][{self.token2wav_infer_device}] token2wav_bigvgan infer time: {(time.perf_counter() - token2wav_bigvgan_start_time)*1000} ms")
        waveform.squeeze().cpu()
        return thinker_result.sequences, waveform.float()
