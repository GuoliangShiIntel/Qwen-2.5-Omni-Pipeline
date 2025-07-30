"""
Qwen2.5 Omni Utility Functions Module
Contains utility functions and classes extracted from qwen2_5_omni_helper.py
"""

import torch
import numpy as np
from typing import Optional, Union, List


# Model file name constants
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
    t_index: List[int],
    grid_hs: List[int],
    grid_ws: List[int],
) -> torch.Tensor:
    """
    Generate LLM position IDs for vision modality
    
    Args:
        start_idx: Starting index
        vision_idx: Vision index
        spatial_merge_size: Spatial merge size
        t_index: Time index list
        grid_hs: Grid height list
        grid_ws: Grid width list
    
    Returns:
        torch.Tensor: Position ID tensor
    """
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def get_chunked_index(llm_pos_ids: torch.Tensor, t_ntoken_per_chunk: int, st_idx: int) -> List[tuple]:
    """
    Get chunked indices
    
    Args:
        llm_pos_ids: LLM position ID tensor
        t_ntoken_per_chunk: Number of tokens per chunk
        st_idx: Starting index
    
    Returns:
        List[tuple]: List of chunk indices, each element is a (start, end) tuple
    """
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
    """
    Calculate RoPE (Rotary Position Embedding) indices
    
    Args:
        config: Model configuration
        input_ids: Input ID tensor
        image_grid_thw: Image grid time, height, width
        video_grid_thw: Video grid time, height, width
        attention_mask: Attention mask
        use_audio_in_video: Whether to use audio in video
        audio_seqlens: Audio sequence lengths
        second_per_grids: Seconds per grid
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Position IDs and position deltas
    """
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
            remain_images, remain_videos, remain_audios = (
                image_nums,
                video_nums,
                audio_nums,
            )
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


class RungeKutta4ODESolver:
    """
    Fourth-order Runge-Kutta ODE solver
    Used for numerical solution of initial value problems for ordinary differential equations
    """
    
    def __init__(self, function, initial_value):
        """
        Initialize the solver
        
        Args:
            function: Differential equation function f(t, y)
            initial_value: Initial value
        """
        self.function = function
        self.initial_value = initial_value

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_step(self, function, time_start, time_step, time_end, value_start, function_value_start=None):
        """
        Execute one step of fourth-order Runge-Kutta calculation
        
        Args:
            function: Differential equation function
            time_start: Start time
            time_step: Time step
            time_end: End time
            value_start: Start value
            function_value_start: Function value at start point (optional, for optimization)
        
        Returns:
            Calculated increment value
        """
        k1 = function_value_start if function_value_start is not None else function(time_start, value_start)
        k2 = function(time_start + time_step * self._one_third, value_start + time_step * k1 * self._one_third)
        k3 = function(time_start + time_step * self._two_thirds, value_start + time_step * (k2 - k1 * self._one_third))
        k4 = function(time_end, value_start + time_step * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * time_step / 8

    def _compute_step(self, function, time_start, time_step, time_end, value_start):
        """
        Compute the solution for one time step
        
        Args:
            function: Differential equation function
            time_start: Start time
            time_step: Time step
            time_end: End time
            value_start: Start value
        
        Returns:
            tuple: (increment value, function value at start point)
        """
        function_value_start = function(time_start, value_start)
        return self._rk4_step(function, time_start, time_step, time_end, value_start, function_value_start=function_value_start), function_value_start

    def _linear_interpolation(self, time_start, time_end, value_start, value_end, time_point):
        """
        Linear interpolation
        
        Args:
            time_start: Start time
            time_end: End time
            value_start: Start value
            value_end: End value
            time_point: Interpolation point time
        
        Returns:
            Interpolation result
        """
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points):
        """
        Solve the differential equation at given time point sequence
        
        Args:
            time_points: Time point sequence
        
        Returns:
            torch.Tensor: Solution at each time point
        """
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
