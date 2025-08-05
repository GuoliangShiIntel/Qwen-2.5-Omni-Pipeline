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
from typing import Optional, Union, Any, Dict, Tuple

import compile_ov_model_helper as ov_compiler
from qwen2_5_omni_utils import (
    get_llm_pos_ids_for_vision,
    get_chunked_index,
    get_rope_index,
    RungeKutta4ODESolver,
    THINKER_EMBEDDING_NAME,
    THINKER_AUDIO_NAME,
    THINKER_AUDIO_STATE_NAME,
    THINKER_AUDIO_EMBED_NAME,
    THINKER_PATCHER_NAME,
    THINKER_MERGER_NAME,
    THINKER_LANGUAGE_NAME,
    TALKER_LANGUAGE_NAME,
    TALKER_EMBEDDING_NAME,
    TOKEN2WAV_DIT_NAME,
    TOKEN2WAV_BIGVGAN_NAME,
)

core = ov.Core()


# ============================================================================
# Base Processing Classes
# ============================================================================

class BaseModalityProcessor:
    """Base class for modality processors"""
    
    def __init__(self, device: str, config):
        self.device = device
        self.config = config
        
    def process(self, *args, **kwargs):
        """Process the modality input"""
        raise NotImplementedError


class ThinkerTextProcessor(BaseModalityProcessor):
    """Text processing for Thinker module"""
    
    def __init__(self, embed_tokens_model, device: str, config):
        super().__init__(device, config)
        self.embed_tokens = embed_tokens_model
        
    def process_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract text embeddings from input_ids"""
        return torch.from_numpy(self.embed_tokens(input_ids)[0])


class ThinkerAudioProcessor(BaseModalityProcessor):
    """Audio processing for Thinker module"""
    
    def __init__(self, audio_embed_model, audio_model, audio_state_model, device: str, config):
        super().__init__(device, config)
        self.audio_embed = audio_embed_model
        self.audio = audio_model
        self.audio_state = audio_state_model
        self.n_window = config.audio_config.n_window
        
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def _padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
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
        
    def process_audio_features(self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor) -> torch.Tensor:
        """Process audio features through the audio pipeline"""
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        
        audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(audio_feature_lengths)
        feature_lens = audio_feature_lengths
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
        padded_feature, padded_mask, padded_mask_after_cnn = self._padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        
        # Audio embedding processing
        audio_embed_start_time = time.perf_counter()
        padded_embed = torch.from_numpy(self.audio_embed([padded_feature, padded_mask])[0])
        print(f"[Thinker][Audio_ID_0][{self.device}] audio embed infer time: {(time.perf_counter() - audio_embed_start_time)*1000} ms")
        
        hidden_states = padded_embed[padded_mask_after_cnn]
        
        # Audio processing
        audio_start_time = time.perf_counter()
        hidden_states = torch.from_numpy(self.audio([hidden_states, padded_mask_after_cnn])[0])
        print(f"[Thinker][Audio_ID_1][{self.device}] audio infer time: {(time.perf_counter() - audio_start_time)*1000} ms")
        
        # Audio state processing
        hidden_states_list = hidden_states.split(audio_feat_lengths.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            audio_state_start_time = time.perf_counter()
            each_audio_states = torch.from_numpy(self.audio_state([each_audio_states])[0])
            print(f"[Thinker][Audio_ID_2][{self.device}] audio_state infer time: {(time.perf_counter() - audio_state_start_time)*1000} ms")
            token_audio_list.append(each_audio_states)
        
        audio_features = torch.cat(token_audio_list, dim=0)
        
        if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
            raise ValueError("length of audio_features should match audio_output_lengths")
            
        return audio_features, audio_output_lengths


class ThinkerVisionProcessor(BaseModalityProcessor):
    """Vision processing for Thinker module"""
    
    def __init__(self, visual_patcher_model, visual_merger_model, device: str, config):
        super().__init__(device, config)
        self.visual_patcher = visual_patcher_model
        self.visual_merger = visual_merger_model
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size
        self.fullatt_block_indexes = config.vision_config.fullatt_block_indexes
        self.window_size = config.vision_config.window_size
        self.spatial_merge_unit = config.vision_config.spatial_merge_size * config.vision_config.spatial_merge_size
        
        # Initialize rotary position embedding
        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        self._rotary_pos_emb = self._create_rotary_embedding(head_dim // 2)
        
    def _create_rotary_embedding(self, dim: int, theta: float = 10000.0):
        """Create rotary position embedding"""
        class Qwen2_5_VisionRotaryEmbedding(torch.nn.Module):
            def __init__(self, dim: int, theta: float = 10000.0) -> None:
                super().__init__()
                inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            def forward(self, seqlen: int) -> torch.Tensor:
                seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(seq, self.inv_freq)
                return freqs
        
        return Qwen2_5_VisionRotaryEmbedding(dim, theta)
        
    def _rot_pos_emb(self, grid_thw):
        """Calculate rotary position embeddings"""
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

    def _get_window_index(self, grid_thw):
        """Get window indexing for vision processing"""
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
        
    def process_visual_features(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Process visual features through the vision pipeline"""
        # Visual patcher processing
        visual_patcher_start_time = time.perf_counter()
        hidden_states = self.visual_patcher(pixel_values)[0]
        print(f"[Thinker][Vision_ID_0][CPU] visual patcher infer time: {(time.perf_counter() - visual_patcher_start_time)*1000} ms")
        
        # Calculate rotary position embeddings and window indices
        rotary_pos_emb = self._rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self._get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        
        # Create attention masks
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

        # Visual merger processing
        visual_merger_start_time = time.perf_counter()
        res = self.visual_merger([hidden_states, causal_mask, window_causal_mask, window_index, rotary_pos_emb])[0]
        print(f"[Thinker][Vision_ID_1][{self.device}] visual merger infer time: {(time.perf_counter() - visual_merger_start_time)*1000} ms")
        
        return torch.from_numpy(res)

# ============================================================================
# Thinker Module
# ============================================================================

class OVQwen2_5OmniThinkerForConditionalGeneration(GenerationMixin):
    def __init__(self, model_dir, device, config, max_prompt_len=1024, min_response_len=256):
        self.infer_device = device
        self.max_prompt_len = max_prompt_len
        self.min_response_len = min_response_len
        self.model = core.read_model(model_dir / THINKER_LANGUAGE_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}

        # Initialize LLM model
        self._initialize_llm_model(model_dir, device)
        
        # Initialize processors
        self._initialize_processors(model_dir, device, config)
        
        # Initialize configuration
        self._initialize_config(config)
        
        self.llm_times = []

    def _initialize_llm_model(self, model_dir, device):
        """Initialize the LLM model based on device type"""
        if device == "NPU":
            llm_blob_cache_path = model_dir / ".blob_cache" / f"thinker_language_{device}_prompt{self.max_prompt_len}_response{self.min_response_len}.blob"
            weights_bin = model_dir / "openvino_thinker_language_model.bin"
            llm = ov_compiler.npu_llm_model_import_or_compile(
                llm_blob_cache_path, model_dir / THINKER_LANGUAGE_NAME, 
                weights_bin, device, 'thinker_language', 
                max_prompt_len=self.max_prompt_len, min_response_len=self.min_response_len
            )
        else:
            llm_blob_cache_path = model_dir / ".blob_cache" / f"thinker_language_{device}.blob"
            llm = ov_compiler.cpu_gpu_model_import_or_compile(
                llm_blob_cache_path, model_dir / THINKER_LANGUAGE_NAME, 
                device, 'thinker_language', False
            )
        self.request = llm.create_infer_request()
        
    def _initialize_processors(self, model_dir, device, config):
        """Initialize modality processors"""
        # Audio models
        if device == "NPU":
            audio_embed = ov_compiler.npu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_audio_embedding_{device}.blob",
                model_dir / THINKER_AUDIO_EMBED_NAME,
                ov_compiler.convert_thinker_audio_embedding_to_static_shape,
                device, 'thinker_audio_embedding'
            )
            audio = ov_compiler.npu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_audio_{device}.blob",
                model_dir / THINKER_AUDIO_NAME,
                ov_compiler.convert_thinker_audio_to_static_shape,
                device, 'thinker_audio'
            )
            audio_state = ov_compiler.npu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_audio_state_{device}.blob",
                model_dir / THINKER_AUDIO_STATE_NAME,
                ov_compiler.convert_thinker_audio_state_to_static_shape,
                device, 'thinker_audio_state'
            )
            visual_merger = ov_compiler.npu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_vision_merger_{device}.blob",
                model_dir / THINKER_MERGER_NAME,
                ov_compiler.convert_thinker_vision_merger_to_static_shape,
                device, 'thinker_vision_merger'
            )
        else:
            audio_embed = ov_compiler.cpu_gpu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_audio_embedding_{device}.blob",
                model_dir / THINKER_AUDIO_EMBED_NAME, device, 'thinker_audio_embedding'
            )
            audio = ov_compiler.cpu_gpu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_audio_{device}.blob",
                model_dir / THINKER_AUDIO_NAME, device, 'thinker_audio'
            )
            audio_state = ov_compiler.cpu_gpu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_audio_state_{device}.blob",
                model_dir / THINKER_AUDIO_STATE_NAME, device, 'thinker_audio_state'
            )
            visual_merger = ov_compiler.cpu_gpu_model_import_or_compile(
                model_dir / ".blob_cache" / f"thinker_vision_merger_{device}.blob",
                model_dir / THINKER_MERGER_NAME, device, 'thinker_vision_merger'
            )

        # Text and vision models (always on CPU)
        embed_tokens = ov_compiler.cpu_gpu_model_import_or_compile(
            model_dir / ".blob_cache" / f"thinker_embedding_CPU.blob",
            model_dir / THINKER_EMBEDDING_NAME, 'CPU', 'thinker_embedding'
        )
        visual_patcher = ov_compiler.cpu_gpu_model_import_or_compile(
            model_dir / ".blob_cache" / f"thinker_vision_CPU.blob",
            model_dir / THINKER_PATCHER_NAME, 'CPU', 'thinker_vision'
        )

        # Initialize processors
        self.text_processor = ThinkerTextProcessor(embed_tokens, 'CPU', config.thinker_config)
        self.audio_processor = ThinkerAudioProcessor(audio_embed, audio, audio_state, device, config.thinker_config)
        self.vision_processor = ThinkerVisionProcessor(visual_patcher, visual_merger, device, config.thinker_config)
        
    def _initialize_config(self, config):
        """Initialize configuration parameters"""
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
        preprocessed_video_embeds=None,
        **kwargs,
    ):
        if past_key_values != ((),):
            past_key_values = None
            
        # Check if this is the first generation (prefill stage)
        is_prefill = cache_position is None or cache_position[0] == 0
        
        # If not in prefill stage and CDPruner is enabled, need special handling for attention_mask
        if not is_prefill and hasattr(self, 'enable_cdpruner') and self.enable_cdpruner and hasattr(self, '_cdpruner_tokens_removed'):
            # Get the number of tokens reduced by CDPruner
            tokens_removed = self._cdpruner_tokens_removed
            
            # If attention_mask needs length adjustment
            if attention_mask is not None:
                expected_length = attention_mask.shape[1] - tokens_removed
                
                # Simply slice the attention_mask to the expected length (assuming all values are 1)
                attention_mask = attention_mask[:, :expected_length]
                
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
            preprocessed_video_embeds=preprocessed_video_embeds,
            **kwargs,
        )
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _process_audio_inputs(self, inputs_embeds, input_ids, input_features, feature_attention_mask):
        """Process audio features and integrate them into input embeddings"""
        if input_features is None:
            return inputs_embeds
            
        audio_features, audio_output_lengths = self.audio_processor.process_audio_features(
            input_features, feature_attention_mask
        )
        audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(audio_mask, audio_features)
    
    def _process_image_inputs(self, inputs_embeds, input_ids, pixel_values, image_grid_thw):
        """Process image features and integrate them into input embeddings"""
        if pixel_values is None:
            return inputs_embeds
            
        num_images = image_grid_thw.shape[0]
        results = []
        current_index = 0
        
        for i in range(num_images):
            h, w = image_grid_thw[i][1].item(), image_grid_thw[i][2].item()
            image_size = h * w
            image_pixels = pixel_values[current_index:current_index + image_size]
            
            image_embed = self.vision_processor.process_visual_features(
                image_pixels, grid_thw=image_grid_thw[i:i+1, :]
            )
            results.append(image_embed)
            current_index += image_size
        
        image_embeds = torch.cat(results, dim=0)
        image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(image_mask, image_embeds)
    
    def _prepare_video_frames(self, pixel_values_videos, video_grid_thw, preprocessed_video_embeds):
        """Prepare video frame embeddings from either raw pixels or preprocessed embeddings"""
        if preprocessed_video_embeds is not None:
            # Use pre-processed video embeddings
            num_frames = video_grid_thw[0][0].item()
            h, w = video_grid_thw[0][1].item(), video_grid_thw[0][2].item()
            tokens_per_frame = (h // self.spatial_merge_size) * (w // self.spatial_merge_size)
            
            video_frame_results = preprocessed_video_embeds.view(num_frames, tokens_per_frame, -1).unbind(0)
            return [frame.contiguous() for frame in video_frame_results]
        else:
            # Process raw video pixels
            num_frames = video_grid_thw[0][0].item()
            video_grid_thw[0][0] = 1  # Temporarily set to 1 for processing
            h, w = video_grid_thw[0][1].item(), video_grid_thw[0][2].item()
            image_size = h * w
            
            results = []
            for i in range(num_frames):
                start, end = i * image_size, (i + 1) * image_size
                pixel_values_video = pixel_values_videos[start:end]
                video_embed = self.vision_processor.process_visual_features(
                    pixel_values_video, grid_thw=video_grid_thw
                )
                results.append(video_embed)
            
            return results
    
    def _apply_cdpruner_if_enabled(self, video_frame_results, inputs_embeds, input_ids):
        """Apply CDPruner to video frames if enabled"""
        if not (hasattr(self, 'enable_cdpruner') and self.enable_cdpruner and hasattr(self, 'cdpruner')):
            return None, None, None
            
        text_embeddings = self._extract_text_embeddings(inputs_embeds, input_ids)
        pruned_features, selected_tokens, pruning_mask = self._apply_video_pruning(video_frame_results, text_embeddings)
        
        return pruned_features, selected_tokens, pruning_mask
    
    def _integrate_video_embeddings(self, inputs_embeds, input_ids, video_frame_results, pruned_features, pruning_mask):
        """Integrate video embeddings into input embeddings with or without pruning"""
        if pruned_features is not None:
            # CDPruner is enabled - use pruned features
            num_frames = len(video_frame_results)
            original_tokens_per_frame = video_frame_results[0].shape[0]
            pruned_tokens_per_frame = pruning_mask[0].sum().item()
            
            inputs_embeds, removal_mask = self._reconstruct_inputs_embeds_with_pruned_video(
                inputs_embeds, input_ids, pruned_features, None, pruning_mask,
                original_tokens_per_frame, pruned_tokens_per_frame
            )
            return inputs_embeds, removal_mask
        else:
            # No pruning - standard integration
            video_embeds = torch.cat(video_frame_results, dim=0)
            video_mask = (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            return inputs_embeds.masked_scatter(video_mask, video_embeds), None
    
    def _adjust_masks_and_positions(self, inputs_embeds, attention_mask, position_ids, original_seq_len, removal_mask):
        """Adjust attention mask and position IDs if sequence length changed"""
        if inputs_embeds.shape[1] == original_seq_len:
            return attention_mask.to(inputs_embeds.device) if attention_mask is not None else None, position_ids
        
        # Sequence length changed - record the change and adjust masks
        tokens_removed = original_seq_len - inputs_embeds.shape[1]
        self._cdpruner_tokens_removed = tokens_removed
        self._cdpruner_adjusted_length = inputs_embeds.shape[1]
        print(f"[DEBUG] CDPruner adjusted sequence length from {original_seq_len} to {inputs_embeds.shape[1]}, removed {tokens_removed} tokens")
        
        # Adjust attention_mask
        if attention_mask is not None:
            if removal_mask is not None:
                keep_mask = ~removal_mask[0]
                attention_mask = attention_mask[:, keep_mask].to(inputs_embeds.device)
                print(f"[DEBUG] Adjusted attention_mask shape: {attention_mask.shape}")
            else:
                # Fallback: create new attention mask
                attention_mask = torch.ones(
                    attention_mask.shape[0], inputs_embeds.shape[1], 
                    dtype=attention_mask.dtype, device=inputs_embeds.device
                )
        
        # Adjust position_ids
        if position_ids is not None:
            if removal_mask is not None:
                keep_mask = ~removal_mask[0]
                position_ids = position_ids[:, :, keep_mask]
            else:
                # Fallback: reconstruct position_ids
                dims, batch_size, _ = position_ids.shape
                new_position_ids = torch.zeros(
                    dims, batch_size, inputs_embeds.shape[1], 
                    dtype=position_ids.dtype, device=position_ids.device
                )
                for dim in range(dims):
                    for batch_idx in range(batch_size):
                        new_position_ids[dim, batch_idx, :] = torch.arange(inputs_embeds.shape[1])
                position_ids = new_position_ids
        
        return attention_mask, position_ids

    def _process_multimodal_inputs(self, inputs_embeds, input_ids, input_features, feature_attention_mask, 
                                 pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, 
                                 attention_mask, position_ids, original_seq_len,
                                 preprocessed_video_embeds=None):
        """Process multimodal inputs and merge them into embeddings
        
        Args:
            inputs_embeds: Input embeddings tensor
            input_ids: Input token IDs
            input_features: Audio feature inputs
            feature_attention_mask: Audio feature attention mask
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions (time, height, width)
            pixel_values_videos: Video pixel values
            video_grid_thw: Video grid dimensions (time, height, width)
            attention_mask: Attention mask for the sequence
            position_ids: Position IDs for the sequence
            original_seq_len: Original sequence length before processing
            preprocessed_video_embeds: Optional pre-processed video embeddings from external pipeline.
                                     If provided, will skip the video processing with process_visual_features.
                                     
        Returns:
            Tuple containing:
                - inputs_embeds: Processed input embeddings
                - attention_mask: Adjusted attention mask (if sequence length changed)
                - position_ids: Adjusted position IDs (if sequence length changed)
        """
        removal_mask = None
        
        # Process each modality sequentially
        inputs_embeds = self._process_audio_inputs(inputs_embeds, input_ids, input_features, feature_attention_mask)
        inputs_embeds = self._process_image_inputs(inputs_embeds, input_ids, pixel_values, image_grid_thw)
        
        # Process video features if present
        if pixel_values_videos is not None or preprocessed_video_embeds is not None:
            video_frame_results = self._prepare_video_frames(pixel_values_videos, video_grid_thw, preprocessed_video_embeds)
            pruned_features, selected_tokens, pruning_mask = self._apply_cdpruner_if_enabled(video_frame_results, inputs_embeds, input_ids)
            inputs_embeds, removal_mask = self._integrate_video_embeddings(inputs_embeds, input_ids, video_frame_results, pruned_features, pruning_mask)
        
        # Adjust masks and positions if sequence length changed
        attention_mask, position_ids = self._adjust_masks_and_positions(
            inputs_embeds, attention_mask, position_ids, original_seq_len, removal_mask
        )
        
        return inputs_embeds, attention_mask, position_ids

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
        preprocessed_video_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        # Handle audio feature lengths
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
            
        # Handle position IDs and rope deltas
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
                
        # Extract text embeddings if not provided
        if inputs_embeds is None:
            inputs_embeds = self.text_processor.process_text_embeddings(input_ids)
            
        # Process multimodal inputs during prefill stage
        if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
            original_seq_len = inputs_embeds.shape[1]
            
            inputs_embeds, attention_mask, position_ids = self._process_multimodal_inputs(
                inputs_embeds, input_ids, input_features, feature_attention_mask,
                pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw,
                attention_mask, position_ids, original_seq_len,
                preprocessed_video_embeds
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Run LLM inference
        return self._run_llm_inference(inputs_embeds, attention_mask, position_ids, past_key_values, rope_deltas)

    def _run_llm_inference(self, inputs_embeds, attention_mask, position_ids, past_key_values, rope_deltas):
        """Run the LLM inference"""
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
            logits=logits, past_key_values=past_key_values, rope_deltas=rope_deltas, 
            hidden_states=(embeds_to_talker, hidden_states_output)
        )

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
    
    def _extract_text_embeddings(self, inputs_embeds: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Get vision_bos token id (this marks the start of vision content)
        vision_bos_token = getattr(self.config, 'vision_start_token_id', None)
        if vision_bos_token is None:
            vision_bos_token = 151652  # Common value for <|vision_bos|>
        
        # Find the position of <|vision_bos|> token
        vision_bos_positions = (input_ids[0] == vision_bos_token).nonzero(as_tuple=True)[0]
        
        if len(vision_bos_positions) == 0:
            raise ValueError("Can not find vision_bos_positions to extrct test embeddings")
        
        # Take everything before the first <|vision_bos|> as text
        text_end_pos = vision_bos_positions[0].item()
        text_embeddings = inputs_embeds[0, :text_end_pos, :]  # [text_length, feature_dim]
        return text_embeddings
    
    def _validate_video_reconstruction_inputs(self, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, 
                                            pruning_mask: torch.Tensor, original_video_tokens_per_frame: int) -> tuple[int, torch.Tensor]:
        """Validate inputs and extract video token positions"""
        # Ensure batch size is 1
        if inputs_embeds.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {inputs_embeds.shape[0]}")
        if input_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {input_ids.shape[0]}")
        
        # Find all video token positions (video token ID is 151656)
        video_token_id = 151656
        video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
        
        # Validate expected number of video tokens
        num_frames = pruning_mask.shape[0]
        expected_total_tokens = num_frames * original_video_tokens_per_frame
        
        if len(video_positions) != expected_total_tokens:
            raise ValueError(
                f"Expected {expected_total_tokens} video tokens "
                f"(num_frames={num_frames} * tokens_per_frame={original_video_tokens_per_frame}), "
                f"found {len(video_positions)}"
            )
        
        return num_frames, video_positions
    
    def _identify_video_frame_segments(self, video_positions: torch.Tensor, num_frames: int, 
                                     original_video_tokens_per_frame: int) -> list[tuple[int, int]]:
        """Group video positions into frame segments"""
        frame_segments = []
        
        for frame_idx in range(num_frames):
            start_idx = frame_idx * original_video_tokens_per_frame
            end_idx = start_idx + original_video_tokens_per_frame
            
            if end_idx > len(video_positions):
                raise ValueError(f"Not enough video tokens remaining for frame {frame_idx}")
            
            frame_start_pos = video_positions[start_idx].item()
            frame_end_pos = frame_start_pos + original_video_tokens_per_frame
            
            # Verify that this frame's tokens are consecutive
            frame_positions = video_positions[start_idx:end_idx].tolist()
            expected_positions = list(range(frame_start_pos, frame_end_pos))
            
            if frame_positions != expected_positions:
                raise ValueError(
                    f"Frame {frame_idx} tokens are not consecutive. "
                    f"Expected: {expected_positions}, Got: {frame_positions}"
                )
            
            frame_segments.append((frame_start_pos, frame_end_pos))
        
        return frame_segments
    
    def _create_removal_mask(self, inputs_embeds: torch.Tensor, frame_segments: list[tuple[int, int]], 
                           pruning_mask: torch.Tensor) -> torch.Tensor:
        """Create removal mask based on pruning decisions"""
        original_seq_len = inputs_embeds.shape[1]
        removal_mask = torch.zeros(1, original_seq_len, dtype=torch.bool, device=inputs_embeds.device)
        
        for frame_idx, (frame_start, frame_end) in enumerate(frame_segments):
            frame_pruning_mask = pruning_mask[frame_idx]  # [tokens_per_frame]
            tokens_to_remove = ~frame_pruning_mask  # Invert mask to get removed tokens
            
            frame_positions_in_seq = torch.arange(frame_start, frame_end, device=removal_mask.device)
            removal_mask[0, frame_positions_in_seq] = tokens_to_remove
        
        return removal_mask
    
    def _reconstruct_embeddings_with_pruned_frames(self, inputs_embeds: torch.Tensor, frame_segments: list[tuple[int, int]], 
                                                 pruned_video_embeds: torch.Tensor, pruned_video_tokens_per_frame: int) -> torch.Tensor:
        """Reconstruct input embeddings by replacing video frames with pruned versions"""
        new_embeds_parts = []
        current_pos = 0
        
        # Flatten pruned embeddings for efficient indexing
        flattened_pruned_embeds = pruned_video_embeds.view(-1, pruned_video_embeds.shape[-1])
        pruned_embed_idx = 0
        
        for frame_start, frame_end in frame_segments:
            # Add content before this frame
            if current_pos < frame_start:
                new_embeds_parts.append(inputs_embeds[0, current_pos:frame_start, :])
            
            # Add pruned embeddings for this frame
            frame_pruned_embeds = flattened_pruned_embeds[
                pruned_embed_idx:pruned_embed_idx + pruned_video_tokens_per_frame
            ]
            new_embeds_parts.append(frame_pruned_embeds)
            pruned_embed_idx += pruned_video_tokens_per_frame
            
            current_pos = frame_end
        
        # Add remaining content after the last frame
        original_seq_len = inputs_embeds.shape[1]
        if current_pos < original_seq_len:
            new_embeds_parts.append(inputs_embeds[0, current_pos:, :])
        
        # Concatenate all parts
        if new_embeds_parts:
            new_inputs_embeds = torch.cat(new_embeds_parts, dim=0).unsqueeze(0)
        else:
            # Edge case: no content
            new_inputs_embeds = torch.empty(
                1, 0, inputs_embeds.shape[-1], 
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
        
        return new_inputs_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    def _reconstruct_inputs_embeds_with_pruned_video(self, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, 
                                                   pruned_video_embeds: torch.Tensor, selected_tokens: torch.Tensor,
                                                   pruning_mask: torch.Tensor, original_video_tokens_per_frame: int,
                                                   pruned_video_tokens_per_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct inputs_embeds with pruned video embeddings using per-frame pruning information
        
        Args:
            inputs_embeds: Input embeddings tensor [batch_size, seq_len, feature_dim]
            input_ids: Input token IDs [batch_size, seq_len]
            pruned_video_embeds: Pruned video embeddings [num_frames, pruned_token_per_frame, feature_dim]
            selected_tokens: Selected token indices per frame [num_frames, selected_tokens_per_frame]
            pruning_mask: Boolean mask indicating which tokens were kept per frame [num_frames, tokens_per_frame]
            original_video_tokens_per_frame: Number of tokens per frame before pruning (int)
            pruned_video_tokens_per_frame: Number of tokens per frame after pruning (int)
            
        Returns:
            tuple containing:
                - torch.Tensor: Reconstructed input embeddings with pruned video tokens [batch_size, new_seq_len, feature_dim]
                - torch.Tensor: Removal mask indicating which original tokens were removed [batch_size, original_seq_len]
            
        Note:
            Video tokens are organized as contiguous blocks per frame, but frames are separated by other content.
            This function uses per-frame pruning information to precisely reconstruct the embedding sequence
            with the correct pruned tokens for each video frame.
        """
        # Step 1: Validate inputs and extract video positions
        num_frames, video_positions = self._validate_video_reconstruction_inputs(
            inputs_embeds, input_ids, pruning_mask, original_video_tokens_per_frame
        )
        
        # Step 2: Identify frame segments
        frame_segments = self._identify_video_frame_segments(
            video_positions, num_frames, original_video_tokens_per_frame
        )
        
        # Step 3: Create removal mask
        removal_mask = self._create_removal_mask(inputs_embeds, frame_segments, pruning_mask)
        
        # Step 4: Reconstruct embeddings with pruned frames
        new_inputs_embeds = self._reconstruct_embeddings_with_pruned_frames(
            inputs_embeds, frame_segments, pruned_video_embeds, pruned_video_tokens_per_frame
        )
        
        return new_inputs_embeds, removal_mask
    
    def _apply_video_pruning(self, video_frame_results: list, text_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CDPruner to video embeddings with per-frame pruning
        
        Args:
            video_frame_results: List of video frame embeddings [tokens_per_frame, feature_dim]
            text_embeddings: Text embeddings for relevance calculation [text_length, feature_dim]
            
        Returns:
            tuple containing:
                - pruned_features: Pruned video embeddings [total_pruned_tokens, feature_dim]
                - selected_tokens: Selected token indices [B, T]
                - pruning_mask: Boolean mask indicating which tokens were kept [B, N]
        """
        # Stack all video frames: [num_frames, tokens_per_frame, feature_dim]
        video_batch = torch.stack(video_frame_results, dim=0)
        
        # Apply CDPruner to get pruned features and selection information
        pruned_features, selected_tokens, pruning_mask = self.cdpruner.prune_tokens(video_batch, text_embeddings)
        
        # Convert List formats to torch.Tensor if needed
        # selected_tokens: List[List[int]] -> torch.Tensor [B, T]
        if isinstance(selected_tokens, list):
            # Pad sequences to same length and convert to tensor
            max_length = max(len(seq) for seq in selected_tokens) if selected_tokens else 0
            padded_selected_tokens = []
            for seq in selected_tokens:
                padded_seq = seq + [-1] * (max_length - len(seq))  # Pad with -1
                padded_selected_tokens.append(padded_seq)
            selected_tokens = torch.tensor(padded_selected_tokens, dtype=torch.long, device=pruned_features.device)
        
        # pruning_mask: List[bool] -> torch.Tensor [B, N]
        if isinstance(pruning_mask, list):
            pruning_mask = torch.tensor(pruning_mask, dtype=torch.bool, device=pruned_features.device)
            
            # Reshape from [B*N] to [B, N] where B is number of frames, N is tokens per frame
            B = video_batch.shape[0]  # number of frames
            N = len(pruning_mask) // B  # tokens per frame
            pruning_mask = pruning_mask.view(B, N)
        
        return pruned_features, selected_tokens, pruning_mask
    
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


# ============================================================================
# Talker Module  
# ============================================================================

class TalkerProcessor:
    """Talker input processing utilities"""
    
    @staticmethod
    def _get_pytorch_device(openvino_device: str) -> str:
        """Convert OpenVINO device to PyTorch-compatible device"""
        if openvino_device == "NPU":
            return "cpu"  # NPU models use CPU tensors as input
        elif openvino_device == "GPU":
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return openvino_device  # CPU and other devices
    
    @staticmethod
    def prepare_talker_inputs(thinker_result, input_ids, speaker_params, talker_device, thinker_embed_tokens):
        """Prepare inputs for talker from thinker results"""
        # Convert OpenVINO device to PyTorch device for tensor operations
        pytorch_device = TalkerProcessor._get_pytorch_device(talker_device)
        
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(pytorch_device)
        thinker_token_embeds = [x[0].to(pytorch_device) for x in thinker_result.hidden_states]
        thinker_hidden_states = [x[1].to(pytorch_device) for x in thinker_result.hidden_states]
        
        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(pytorch_device),
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=pytorch_device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        return {
            'thinker_generate_ids': thinker_generate_ids,
            'thinker_token_embeds': thinker_token_embeds,
            'thinker_hidden_states': thinker_hidden_states,
            'talker_input_text_ids': talker_input_text_ids,
            'talker_text_bos_token': talker_text_bos_token
        }
    
    @staticmethod
    def prepare_talker_embeddings(talker_data, codec_tokens, thinker_embed_tokens, talker_device):
        """Prepare embeddings for talker"""
        # Convert OpenVINO device to PyTorch device for tensor operations
        pytorch_device = TalkerProcessor._get_pytorch_device(talker_device)
        
        thinker_reply_part = torch.cat(talker_data['thinker_hidden_states'][1:], dim=1) + torch.cat(talker_data['thinker_token_embeds'][1:], dim=1)
        talker_inputs_embeds = talker_data['thinker_hidden_states'][0] + talker_data['thinker_token_embeds'][0]
        
        # Text embeddings always run on CPU device
        talker_text_bos_token = torch.tensor([[talker_data['talker_text_bos_token']]], dtype=torch.long, device='cpu')
        talker_text_bos_embed = torch.from_numpy(thinker_embed_tokens(talker_text_bos_token)[0]).to(pytorch_device)

        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )
        
        return talker_inputs_embeds, thinker_reply_part


class OVQwen2_5OmniTalkerForConditionalGeneration(GenerationMixin):
    def __init__(self, model_dir, device, config, max_prompt_len=1024, min_response_len=256):
        self.max_prompt_len = max_prompt_len
        self.min_response_len = min_response_len
        self.infer_device = device
        self.model = core.read_model(model_dir / TALKER_LANGUAGE_NAME)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}

        # Initialize LLM model
        self._initialize_llm_model(model_dir, device)
        
        # Initialize text embedding
        text_embedding_blob_cache_path = model_dir / ".blob_cache" / f"talker_embedding_CPU.blob"
        self.embed_tokens = ov_compiler.cpu_gpu_model_import_or_compile(text_embedding_blob_cache_path, model_dir / TALKER_EMBEDDING_NAME, 'CPU', 'talker_embedding')

        # Initialize configuration
        self._initialize_config(config)
        
        self.llm_times = []

    def _initialize_llm_model(self, model_dir, device):
        """Initialize the LLM model based on device type"""
        if device == "NPU":
            llm_blob_cache_path = model_dir / ".blob_cache" / f"talker_language_{device}_prompt{self.max_prompt_len}_response{self.min_response_len}.blob"
            weights_bin = model_dir / "openvino_talker_language_model.bin"
            llm = ov_compiler.npu_llm_model_import_or_compile(
                llm_blob_cache_path, model_dir / TALKER_LANGUAGE_NAME, 
                weights_bin, device, 'talker_language', 
                max_prompt_len=self.max_prompt_len, min_response_len=self.min_response_len
            )
        else:
            llm_blob_cache_path = model_dir / ".blob_cache" / f"talker_language_{device}.blob"
            llm = ov_compiler.cpu_gpu_model_import_or_compile(
                llm_blob_cache_path, model_dir / TALKER_LANGUAGE_NAME, 
                device, 'talker_language', False
            )
        self.request = llm.create_infer_request()

    def _initialize_config(self, config):
        """Initialize configuration parameters"""
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
        # Handle position IDs and rope deltas
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

        # Prepare embeddings
        if inputs_embeds is None:
            codec_embeds = torch.from_numpy(self.embed_tokens(input_ids)[0])
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]
                
        # Run LLM inference
        return self._run_llm_inference(inputs_embeds, attention_mask, position_ids, past_key_values, rope_deltas, thinker_reply_part)

    def _run_llm_inference(self, inputs_embeds, attention_mask, position_ids, past_key_values, rope_deltas, thinker_reply_part):
        """Run the LLM inference"""
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

    def _get_initial_cache_position(self, input_ids, device, model_kwargs):
        # Talker needs to calculate cache_position with input_ids, so pop inputs_embeds temporarily
        inputs_embeds = model_kwargs.pop("inputs_embeds")
        model_kwargs = super()._get_initial_cache_position(input_ids, device, model_kwargs)
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


# ============================================================================
# Token2Wav Module
# ============================================================================

class Token2WavProcessor:
    """Token2Wav processing utilities"""
    
    def __init__(self, token2wav_dit_model, token2wav_bigvgan_model, device: str, config):
        self.token2wav_dit = token2wav_dit_model
        self.token2wav_bigvgan = token2wav_bigvgan_model
        self.device = device
        self.config = config
        
    def process_codes_to_audio(self, talker_generate_codes: torch.Tensor, speaker_params: Dict) -> torch.Tensor:
        """Convert talker generated codes to audio waveform"""
        original_length = talker_generate_codes.shape[1]
        chunk_size = 128
        
        print(f"[Token2wav][{self.device}] Processing {original_length} tokens in chunks of {chunk_size}")
        
        # Split codes into chunks of 128 tokens
        waveform_chunks = []
        num_chunks = (original_length + chunk_size - 1) // chunk_size  # Ceiling division
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, original_length)
            
            # Extract current chunk
            current_chunk = talker_generate_codes[:, start_idx:end_idx]
            current_chunk_length = current_chunk.shape[1]
            
            # Only NPU needs padding for incomplete chunks
            needs_padding = current_chunk_length < chunk_size and self.device == "NPU"
            
            if needs_padding:
                padding_length = chunk_size - current_chunk_length
                padding = torch.zeros((current_chunk.shape[0], padding_length), 
                                    dtype=current_chunk.dtype, 
                                    device=current_chunk.device)
                current_chunk = torch.cat([current_chunk, padding], dim=1)

            # Process current chunk
            chunk_waveform = self._process_single_chunk(current_chunk, speaker_params, chunk_idx + 1, num_chunks, current_chunk_length if needs_padding else None)
            waveform_chunks.append(chunk_waveform)
        
        # Concatenate all waveform chunks
        if len(waveform_chunks) == 1:
            final_waveform = waveform_chunks[0]
        else:
            final_waveform = torch.cat(waveform_chunks, dim=-1)  # Concatenate along time dimension
        
        print(f"[Token2wav][{self.device}] Final waveform shape: {final_waveform.shape}")
        return final_waveform.squeeze().cpu().float()

    def _process_single_chunk(self, talker_generate_codes: torch.Tensor, speaker_params: Dict, chunk_idx: int, total_chunks: int, original_length: int = None) -> torch.Tensor:
        """Process a single chunk of tokens to generate audio waveform
        
        Args:
            talker_generate_codes: Input codes tensor (may be padded for NPU)
            speaker_params: Speaker parameters
            chunk_idx: Current chunk index (1-based)
            total_chunks: Total number of chunks
            original_length: Original length before padding (for NPU only)
        """
        # Prepare parameters
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

        # Create ODE function
        def ode_function(time_step, hidden_states):
            token2wav_dit_start_time = time.perf_counter()
            model_output = torch.from_numpy(
                self.token2wav_dit([hidden_states, reference_mel_spectrogram, conditioning_vector, talker_generate_codes, time_step])[0]
            )
            print(f"[Token2wav][Model_ID_0][{self.device}] Chunk {chunk_idx}/{total_chunks} token2wav_dit infer time: {(time.perf_counter() - token2wav_dit_start_time)*1000} ms")
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        # Solve ODE
        initial_time = 0
        time_embedding = torch.linspace(initial_time, 1, 10, device=talker_generate_codes.device, dtype=conditioning_vector.dtype)

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        print(f"[Token2wav][Model_ID_0] Chunk {chunk_idx}/{total_chunks} hidden_states Shape {initial_state.shape}")
        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        # Generate final waveform
        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        print(f"[Token2wav][Model_ID_1] Chunk {chunk_idx}/{total_chunks} mel_spectrogram Shape {generated_mel_spectrogram.shape}")
        
        token2wav_bigvgan_start_time = time.perf_counter()
        waveform = torch.from_numpy(self.token2wav_bigvgan([generated_mel_spectrogram])[0])
        print(f"[Token2wav][Model_ID_1][{self.device}] Chunk {chunk_idx}/{total_chunks} token2wav_bigvgan infer time: {(time.perf_counter() - token2wav_bigvgan_start_time)*1000} ms")
        
        # For NPU, trim the output to match original length if padding was used
        if original_length is not None and self.device == "NPU":
            # Calculate how much to trim based on the ratio of original to padded length
            actual_length = talker_generate_codes.shape[1]
            trim_ratio = original_length / actual_length
            target_samples = int(waveform.shape[-1] * trim_ratio)
            waveform = waveform[..., :target_samples]
        
        return waveform


# ============================================================================
# Main Model Pipeline
# ============================================================================


class OVQwen2_5OmniModel(GenerationMixin):
    def __init__(self, model_dir, thinker_device, talker_device, token2wav_device, enable_talker, 
                 thinker_max_prompt_len=1024, thinker_min_response_len=256,
                 talker_max_prompt_len=1024, talker_min_response_len=256,
                 enable_cdpruner=False, cdpruner_num_visual_tokens=256, cdpruner_relevance_weight=0.5):
        """Initialize OVQwen2_5OmniModel with separate parameters for each LLM
        
        Args:
            model_dir: Path to model directory
            thinker_device: Device for thinker LLM (CPU/GPU/NPU)
            talker_device: Device for talker LLM (CPU/GPU/NPU)
            token2wav_device: Device for token2wav models (CPU/GPU/NPU)
            enable_talker: Whether to enable talker functionality
            thinker_max_prompt_len: Maximum prompt length for thinker LLM (default: 1024)
            thinker_min_response_len: Minimum response length for thinker LLM (default: 256)
            talker_max_prompt_len: Maximum prompt length for talker LLM (default: 1024)
            talker_min_response_len: Minimum response length for talker LLM (default: 256)
            enable_cdpruner: Whether to enable CDPruner for video token pruning (default: False)
            cdpruner_num_visual_tokens: Number of visual tokens to keep after pruning (default: 256)
            cdpruner_relevance_weight: Weight for balancing relevance vs diversity (default: 0.5)
        """
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.thinker_infer_device = thinker_device
        self.talker_infer_device = talker_device
        
        # CDPruner configuration
        self.enable_cdpruner = enable_cdpruner
        self.cdpruner_num_visual_tokens = cdpruner_num_visual_tokens
        if self.enable_cdpruner:
            from cdpruner.cdpruner import CDPruner
            from cdpruner.cdpruner_config import Config as CDPrunerConfig
            # Use the specified number of visual tokens directly
            cdpruner_config = CDPrunerConfig(
                num_visual_tokens=cdpruner_num_visual_tokens,
                relevance_weight=cdpruner_relevance_weight,
                enable_pruning=True,
                device="CPU",
                debug_mode=True
            )
            self.cdpruner = CDPruner(cdpruner_config)
        self.token2wav_infer_device = token2wav_device
        
        # Store parameters for each LLM separately
        self.thinker_max_prompt_len = thinker_max_prompt_len
        self.thinker_min_response_len = thinker_min_response_len
        self.talker_max_prompt_len = talker_max_prompt_len
        self.talker_min_response_len = talker_min_response_len
        
        self.has_talker = enable_talker
        
        model_path = Path(model_dir)
        
        # Initialize Thinker
        self.thinker = OVQwen2_5OmniThinkerForConditionalGeneration(
            model_path / "thinker", thinker_device, self.config, 
            self.thinker_max_prompt_len, self.thinker_min_response_len
        )
        
        # Configure CDPruner in thinker
        self.thinker.enable_cdpruner = self.enable_cdpruner
        if self.enable_cdpruner:
            self.thinker.cdpruner = self.cdpruner
            self.thinker.cdpruner_num_visual_tokens = self.cdpruner_num_visual_tokens
        
        # Initialize Talker and Token2Wav if enabled
        if self.has_talker:
            self.enable_talker(model_path, talker_device, token2wav_device)
            
        # Load speaker mappings
        self.speaker_map = {}
        spk_path = model_path / "spk_dict.pt"
        self.load_speakers(spk_path)

    def enable_talker(self, model_path, talker_device, token2wav_device=None):
        """Enable talker and token2wav functionality"""
        if token2wav_device is None:
            token2wav_device = talker_device
            
        # Initialize Talker with its own parameters
        self.talker = OVQwen2_5OmniTalkerForConditionalGeneration(
            model_path / "talker", talker_device, self.config, 
            self.talker_max_prompt_len, self.talker_min_response_len
        )

        # Initialize Token2Wav models
        if token2wav_device == 'NPU':
            token2wav_dit = ov_compiler.npu_model_import_or_compile(
                model_path / ".blob_cache" / f"token2wav_dit_{token2wav_device}.blob",
                model_path / TOKEN2WAV_DIT_NAME,
                ov_compiler.convert_token2wav_dit_to_static_shape,
                token2wav_device, 'token2wav_dit'
            )
            token2wav_bigvgan = ov_compiler.npu_model_import_or_compile(
                model_path / ".blob_cache" / f"token2wav_bigvgan_{token2wav_device}.blob",
                model_path / TOKEN2WAV_BIGVGAN_NAME,
                ov_compiler.convert_token2wav_bigvgan_to_static_shape,
                token2wav_device, 'token2wav_bigvgan'
            )
        else:
            token2wav_dit = ov_compiler.cpu_gpu_model_import_or_compile(
                model_path / ".blob_cache" / f"token2wav_dit_{token2wav_device}.blob",
                model_path / TOKEN2WAV_DIT_NAME, token2wav_device, 'token2wav_dit'
            )
            token2wav_bigvgan = ov_compiler.cpu_gpu_model_import_or_compile(
                model_path / ".blob_cache" / f"token2wav_bigvgan_{token2wav_device}.blob",
                model_path / TOKEN2WAV_BIGVGAN_NAME, token2wav_device, 'token2wav_bigvgan'
            )
            
        # Initialize Token2Wav processor
        self.token2wav_processor = Token2WavProcessor(
            token2wav_dit, token2wav_bigvgan, token2wav_device, self.config
        )
        
        self.has_talker = True

    def load_speakers(self, path):
        """Load speaker configurations"""
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value

    def disable_talker(self):
        """Disable talker functionality"""
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav_processor"):
            del self.token2wav_processor
        self.has_talker = False

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def _run_thinker_pipeline(self, input_ids: torch.Tensor, **thinker_kwargs) -> Any:
        """Run the thinker pipeline"""
        print("[===start thinker===]")
        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)
        print(f"[Thinker][LLM] Input Shape: {input_ids.shape}")

        # Log performance metrics
        llm_thinker_times = self.thinker.llm_times
        print(f"[Thinker][LLM_Prefill][{self.thinker_infer_device}] Infer time: {llm_thinker_times[0]*1000} ms")
        remaining_list = llm_thinker_times[1:]
        if remaining_list:
            average = sum(remaining_list) / len(remaining_list)
            print(f"[Thinker][LLM_KV_CACHE][{self.thinker_infer_device}] Infer: {1 / average} token/s")
            
        return thinker_result

    def _run_talker_pipeline(self, thinker_result, input_ids: torch.Tensor, speaker_params: Dict, **talker_kwargs) -> torch.Tensor:
        """Run the talker pipeline"""
        print("[===start talker===]")
        
        # Prepare talker inputs
        talker_data = TalkerProcessor.prepare_talker_inputs(
            thinker_result, input_ids, speaker_params, self.talker.infer_device, self.thinker.text_processor.embed_tokens
        )
        
        # Get PyTorch-compatible device for tensor operations
        pytorch_device = TalkerProcessor._get_pytorch_device(self.talker.infer_device)
        
        # Prepare codec input IDs
        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token, device=pytorch_device),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=pytorch_device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=pytorch_device),
            ],
            dim=1,
        )
        
        # Prepare embeddings
        talker_inputs_embeds, thinker_reply_part = TalkerProcessor.prepare_talker_embeddings(
            talker_data, talker_input_ids, self.thinker.text_processor.embed_tokens, self.talker.infer_device
        )
        
        # Add special tokens to thinker_reply_part
        eos_embedding = torch.from_numpy(
            self.thinker.text_processor.embed_tokens(torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device='cpu'))[0]
        ).to(pytorch_device)

        pad_embedding = torch.from_numpy(
            self.thinker.text_processor.embed_tokens(torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device='cpu'))[0]
        ).to(pytorch_device)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )

        # Prepare attention mask
        talker_attention_mask = None
        if "attention_mask" in talker_kwargs:
            talker_attention_mask = torch.cat([talker_kwargs["attention_mask"], talker_kwargs["attention_mask"].new_ones((1, 2))], dim=1).to(pytorch_device)
            
        print(f"[Talker][LLM] Input Shape: {talker_input_ids.shape}")
        
        # Generate talker output
        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_data['talker_input_text_ids'],
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{k: (v.to(pytorch_device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
        )
        
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]
        print(f"[Talker][LLM] Generate Shape: {talker_generate_codes.shape}")

        # Log performance metrics
        llm_talker_times = self.talker.llm_times
        print(f"[Talker][LLM_Prefill][{self.talker_infer_device}] Infer time: {llm_talker_times[0]*1000} ms")
        remaining_list = llm_talker_times[1:]
        if remaining_list:
            average = sum(remaining_list) / len(remaining_list)
            print(f"[Talker][LLM_KV_CACHE][{self.talker_infer_device}] Infer: {1 / average} token/s")
            
        return talker_generate_codes

    def _run_token2wav_pipeline(self, talker_generate_codes: torch.Tensor, speaker_params: Dict) -> torch.Tensor:
        """Run the token2wav pipeline"""
        print("[===start token2wav===]")
        return self.token2wav_processor.process_codes_to_audio(talker_generate_codes, speaker_params)

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
            speaker (`str` , defaults to "Chelsie"):
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
        # Validate inputs
        if speaker not in self.speaker_map:
            raise ValueError(f"{speaker} is not available, available speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError("Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker.")
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

        # Prepare kwargs for different modules
        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {"max_new_tokens": thinker_max_new_tokens}
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }

        # Process kwargs by prefix
        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            else:
                shared_kwargs[key] = value

        # Merge shared kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value

        speaker_params = self.speaker_map[speaker]

        # Configure for audio generation
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True
        if stream_config is not None:
            thinker_kwargs["streamer"] = stream_config

        # 1. Run Thinker Pipeline
        thinker_result = self._run_thinker_pipeline(input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result

        # 2. Run Talker Pipeline
        talker_generate_codes = self._run_talker_pipeline(thinker_result, input_ids, speaker_params, **talker_kwargs)

        # 3. Run Token2Wav Pipeline
        waveform = self._run_token2wav_pipeline(talker_generate_codes, speaker_params)

        return thinker_result.sequences, waveform
