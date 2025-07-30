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
            llm_blob_cache_path = model_dir / ".blob_cache" / "thinker_language_npuw.blob"
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
                model_dir / ".blob_cache" / "thinker_audio_embedding.blob",
                model_dir / THINKER_AUDIO_EMBED_NAME,
                ov_compiler.convert_thinker_audio_embedding_to_static_shape,
                device, 'thinker_audio_embedding'
            )
            audio = ov_compiler.npu_model_import_or_compile(
                model_dir / ".blob_cache" / "thinker_audio.blob",
                model_dir / THINKER_AUDIO_NAME,
                ov_compiler.convert_thinker_audio_to_static_shape,
                device, 'thinker_audio'
            )
            audio_state = ov_compiler.npu_model_import_or_compile(
                model_dir / ".blob_cache" / "thinker_audio_state.blob",
                model_dir / THINKER_AUDIO_STATE_NAME,
                ov_compiler.convert_thinker_audio_state_to_static_shape,
                device, 'thinker_audio_state'
            )
            visual_merger = ov_compiler.npu_model_import_or_compile(
                model_dir / ".blob_cache" / "thinker_vision_merger.blob",
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

    def _process_multimodal_inputs(self, inputs_embeds, input_ids, input_features, feature_attention_mask, 
                                 pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw):
        """Process multimodal inputs and merge them into embeddings"""
        # Process audio features
        if input_features is not None:
            audio_features, audio_output_lengths = self.audio_processor.process_audio_features(
                input_features, feature_attention_mask
            )
            audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        # Process image features  
        if pixel_values is not None:
            num_images = image_grid_thw.shape[0]
            current_index = 0
            results = []
            for i in range(num_images):
                h = image_grid_thw[i][1].item()
                w = image_grid_thw[i][2].item()
                image_size = h * w
                start = current_index
                end = start + image_size
                image_pixels = pixel_values[start:end]

                image_embed = self.vision_processor.process_visual_features(
                    image_pixels, grid_thw=image_grid_thw[i:i+1, :]
                )
                results.append(image_embed)
                current_index = end

            image_embeds = torch.cat(results, dim=0)
            image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Process video features
        if pixel_values_videos is not None:
            num_images = video_grid_thw[0][0].item()
            video_grid_thw[0][0] = 1
            h = video_grid_thw[0][1].item()
            w = video_grid_thw[0][2].item()
            image_size = h * w
            results = []
            for i in range(num_images):
                start = i * image_size
                end = start + image_size
                pixel_values_video = pixel_values_videos[start:end]
                video_embed = self.vision_processor.process_visual_features(
                    pixel_values_video, grid_thw=video_grid_thw
                )
                results.append(video_embed)

            video_embeds = torch.cat(results, dim=0)
            video_mask = (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
        return inputs_embeds

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
        # Handle audio feature lengths
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
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
            inputs_embeds = self._process_multimodal_inputs(
                inputs_embeds, input_ids, input_features, feature_attention_mask,
                pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw
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
            llm_blob_cache_path = model_dir / ".blob_cache" / "talker_language_npuw.blob"
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
        # Handle device constraints
        if self.device == 'NPU' or self.device == 'GPU':
            current_length = talker_generate_codes.shape[1]
            if current_length > 128:
                talker_generate_codes = talker_generate_codes[:, :128]
                print(f"[{self.device}] Sliced talker_generate_codes from {current_length} to 128")
            elif current_length < 128:
                padding_length = 128 - current_length
                padding = torch.zeros((talker_generate_codes.shape[0], padding_length), 
                                    dtype=talker_generate_codes.dtype, 
                                    device=talker_generate_codes.device)
                talker_generate_codes = torch.cat([talker_generate_codes, padding], dim=1)
                print(f"[{self.device}] Padded talker_generate_codes from {current_length} to 128")

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
            print(f"[Token2wav][Model_ID_0][{self.device}] token2wav_dit infer time: {(time.perf_counter() - token2wav_dit_start_time)*1000} ms")
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        # Solve ODE
        initial_time = 0
        time_embedding = torch.linspace(initial_time, 1, 10, device=talker_generate_codes.device, dtype=conditioning_vector.dtype)

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        print(f"[Token2wav][Model_ID_0][hidden_states] Shape {initial_state.shape}")
        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        # Generate final waveform
        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        print(f"[Token2wav][Model_ID_1][mel_spectrogram] Shape {generated_mel_spectrogram.shape}")
        
        token2wav_bigvgan_start_time = time.perf_counter()
        waveform = torch.from_numpy(self.token2wav_bigvgan([generated_mel_spectrogram])[0])
        print(f"[Token2wav][Model_ID_1][{self.device}] token2wav_bigvgan infer time: {(time.perf_counter() - token2wav_bigvgan_start_time)*1000} ms")
        
        return waveform.squeeze().cpu().float()


# ============================================================================
# Main Model Pipeline
# ============================================================================


class OVQwen2_5OmniModel(GenerationMixin):
    def __init__(self, model_dir, thinker_device, talker_device, token2wav_device, enable_talker, max_prompt_len=1024, min_response_len=256):
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.thinker_infer_device = thinker_device
        self.talker_infer_device = talker_device
        self.token2wav_infer_device = token2wav_device
        self.max_prompt_len = max_prompt_len
        self.min_response_len = min_response_len
        self.has_talker = enable_talker
        
        model_path = Path(model_dir)
        
        # Initialize Thinker
        self.thinker = OVQwen2_5OmniThinkerForConditionalGeneration(
            model_path / "thinker", thinker_device, self.config, max_prompt_len, min_response_len
        )
        
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
            
        # Initialize Talker
        self.talker = OVQwen2_5OmniTalkerForConditionalGeneration(
            model_path / "talker", talker_device, self.config, self.max_prompt_len, self.min_response_len
        )

        # Initialize Token2Wav models
        if token2wav_device == 'NPU':
            token2wav_dit = ov_compiler.npu_model_import_or_compile(
                model_path / ".blob_cache" / "token2wav_dit.blob",
                model_path / TOKEN2WAV_DIT_NAME,
                ov_compiler.convert_token2wav_dit_to_static_shape,
                token2wav_device, 'token2wav_dit'
            )
            token2wav_bigvgan = ov_compiler.npu_model_import_or_compile(
                model_path / ".blob_cache" / "token2wav_bigvgan.blob",
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
