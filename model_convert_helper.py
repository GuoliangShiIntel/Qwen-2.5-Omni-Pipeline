from pathlib import Path
import types
import gc

import openvino as ov
import shutil

try:
    from openvino import opset13
except ImportError:
    from openvino.runtime import opset13
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
import nncf
import numpy as np
import torch
from torch import nn
from transformers import AutoProcessor
from transformers import Qwen2_5OmniForConditionalGeneration
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
)
from pathlib import Path
import types
from typing import Optional, Union, List
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from huggingface_hub import snapshot_download, hf_hub_download

def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model, dim):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[dim:]]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )


core = ov.Core()


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

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


def convert_qwen2_5_omni_model(model_id, output_dir, quantization_config=None, use_local_dir=False):
    thinker_output_dir = Path(output_dir) / "thinker"
    talker_output_dir = Path(output_dir) / "talker"

    thinker_embedding_path = thinker_output_dir / THINKER_EMBEDDING_NAME
    thinker_audio_path = thinker_output_dir / THINKER_AUDIO_NAME
    thinker_audio_embed_path = thinker_output_dir / THINKER_AUDIO_EMBED_NAME
    thinker_audio_state_path = thinker_output_dir / THINKER_AUDIO_STATE_NAME
    thinker_patcher_path = thinker_output_dir / THINKER_PATCHER_NAME
    thinker_merger_path = thinker_output_dir / THINKER_MERGER_NAME
    thinker_lang_path = thinker_output_dir / THINKER_LANGUAGE_NAME

    talker_lang_path = talker_output_dir / TALKER_LANGUAGE_NAME
    talker_embedding_path = talker_output_dir / TALKER_EMBEDDING_NAME

    token2wav_dit_path = output_dir / TOKEN2WAV_DIT_NAME
    token2wav_bigvgan_path = output_dir / TOKEN2WAV_BIGVGAN_NAME
    if all(
        [
            thinker_lang_path.exists(),
            thinker_audio_embed_path.exists(),
            thinker_audio_path.exists(),
            thinker_audio_state_path.exists(),
            thinker_embedding_path.exists(),
            thinker_patcher_path.exists(),
            thinker_merger_path.exists(),
            talker_lang_path.exists(),
            talker_embedding_path.exists(),
            token2wav_dit_path.exists(),
            token2wav_bigvgan_path.exists(),
        ]
    ):
        print(f"✅ {model_id} model already converted. You can find results in {output_dir}")
        return
    print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")

    if use_local_dir:
        ckpt = Path(output_dir) / "ckpt"
        if not ckpt.exists():
            snapshot_download(model_id, local_dir=ckpt, force_download=True)
            shutil.copy(ckpt / "spk_dict.pt", Path(output_dir) / "spk_dict.pt")
    else:
        ckpt = model_id
        if not (Path(output_dir) / "spk_dict.pt").exists():
            hf_hub_download(model_id, filename="spk_dict.pt", local_dir=output_dir)

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.float16)
    model.eval()
    processor = AutoProcessor.from_pretrained(ckpt)

    model.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("✅ Original model successfully loaded")

    if not thinker_embedding_path.exists():
        print("⌛ Convert thinker embedding model")
        __make_16bit_traceable(model.thinker.model.get_input_embeddings())
        ov_model = ov.convert_model(
            model.thinker.model.get_input_embeddings(),
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_model, thinker_embedding_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Thinker embedding model successfully converted")

    def forward_wrap_embedd_audio(self, padded_feature, padded_mask):
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
        return padded_embed

    def forward_wrap_audio(self, hidden_states, padded_mask_after_cnn):
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]
        return hidden_states

    def forward_wrap_audio_state(self, each_audio_states):
        each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
        each_audio_states = self.ln_post(each_audio_states)
        each_audio_states = self.proj(each_audio_states)
        return each_audio_states

    audio = model.thinker.audio_tower
    audio._orig_forward = audio.forward
    if not thinker_audio_path.exists():
        print("⌛ Convert thinker audio model")
        __make_16bit_traceable(audio)
        audio.forward = types.MethodType(forward_wrap_audio, audio)
        ov_model = ov.convert_model(
            audio,
            example_input={
                "hidden_states": torch.randn([1, 1280], dtype=torch.float32),
                "padded_mask_after_cnn": torch.ones([1, 5], dtype=torch.bool),
            },
        )
        ov.save_model(ov_model, thinker_audio_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Thinker audio model successfully converted")
        
    if not thinker_audio_embed_path.exists():
        print("⌛ Convert thinker audio embedding model")
        audio.forward = audio._orig_forward
        audio.forward = types.MethodType(forward_wrap_embedd_audio, audio)
        __make_16bit_traceable(audio)
        ov_model = ov.convert_model(
            audio,
            example_input={
                "padded_feature": torch.randn([1, 128, 9], dtype=torch.float32),
                "padded_mask": torch.ones([1, 1, 9], dtype=torch.int32)
            },
        )
        ov.save_model(ov_model, thinker_audio_embed_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Thinker audio embedding model successfully converted")

    if not thinker_audio_state_path.exists():
        print("⌛ Convert thinker audio state model")
        audio.forward = audio._orig_forward
        audio.forward = types.MethodType(forward_wrap_audio_state, audio)
        __make_16bit_traceable(audio)
        ov_model = ov.convert_model(
            audio,
            example_input={
                "each_audio_states": torch.randn([5, 1280], dtype=torch.float32),
            },
        )
        ov.save_model(ov_model, thinker_audio_state_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Thinker audio state model successfully converted")

    if not thinker_patcher_path.exists() or not thinker_merger_path.exists():
        print("⌛ Convert image embedding model")

        vision_embed_tokens = model.thinker.visual
        if not thinker_patcher_path.exists():
            __make_16bit_traceable(vision_embed_tokens.patch_embed)
            ov_model = ov.convert_model(vision_embed_tokens.patch_embed, example_input={"hidden_states": torch.randn([8, 1176])})
            ov.save_model(ov_model, thinker_patcher_path)
            del ov_model
            cleanup_torchscript_cache()

        def image_embed_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            window_attention_mask: torch.Tensor,
            window_index: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
        ) -> torch.Tensor:
            seq_len = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    attention_mask_now = attention_mask
                else:
                    attention_mask_now = window_attention_mask
                hidden_states = blk(hidden_states, attention_mask=attention_mask_now, rotary_pos_emb=rotary_pos_emb)

            hidden_states = self.merger(hidden_states)
            reverse_indices = torch.argsort(window_index)
            hidden_states = hidden_states[reverse_indices, :]

            return hidden_states

        def sdpa_attn_forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb_vision

            seq_length = hidden_states.shape[0]
            q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
            k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
            v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output

        def block_forward(self, hidden_states, attention_mask, rotary_pos_emb) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
            return hidden_states

        if not thinker_merger_path.exists():
            vision_embed_tokens.forward = types.MethodType(image_embed_forward, vision_embed_tokens)
            for block in vision_embed_tokens.blocks:
                block.forward = types.MethodType(block_forward, block)
                block.attn.forward = types.MethodType(sdpa_attn_forward, block.attn)

            __make_16bit_traceable(vision_embed_tokens)
            ov_model = ov.convert_model(
                vision_embed_tokens,
                example_input={
                    "hidden_states": torch.randn([8, 1280], dtype=torch.float32),
                    "attention_mask": torch.ones([1, 8, 8]),
                    "window_attention_mask": torch.ones([1, 8, 8]),
                    "window_index": torch.ones([2], dtype=torch.int32),
                    "rotary_pos_emb": torch.randn([8, 40]),
                },
            )
            ov.save_model(ov_model, thinker_merger_path)
            del ov_model
            cleanup_torchscript_cache()
        del vision_embed_tokens
        gc.collect()
        print("✅ Image embedding model successfully converted")

    if not thinker_lang_path.exists():
        print("⌛ Convert Thinker Language model")

        def forward_wrap_thinker(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[tuple, BaseModelOutputWithPast]:
            """take care of image_encode, position_ids and (attention_mask = None is fine)"""
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            output = (logits,) + outputs[:]

            return output

        lang_model = model.thinker
        hidden_size = lang_model.model.config.hidden_size
        lang_model._orig_forward = lang_model.forward
        lang_model.forward = types.MethodType(forward_wrap_thinker, lang_model)

        num_pkv = lang_model.model.config.num_hidden_layers
        pkv_shape = (2, lang_model.model.config.num_key_value_heads, 2, hidden_size // lang_model.model.config.num_attention_heads)
        # input_embeds = torch.randn((1, 1, hidden_size))
        cache_position = torch.arange(2, 4)
        position_ids = cache_position.view(1, 1, -1).expand(3, 2, -1)

        input_embeds = torch.randn((2, 2, hidden_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits", "hidden_states"]
        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.append("inputs_embeds")
        example_input = {"inputs_embeds": input_embeds, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": past_key_values}

        input_shapes = [
            ov.PartialShape([-1, -1]),
            ov.PartialShape([3, -1, -1]),
        ]
        input_shapes += (
            [ov.PartialShape([-1, lang_model.model.config.num_key_value_heads, -1, hidden_size // lang_model.model.config.num_attention_heads])] * 2 * num_pkv
        )
        input_shapes += [ov.PartialShape([-1, -1, input_embeds.shape[-1]])]
        __make_16bit_traceable(lang_model)
        ov_model = ov.convert_model(lang_model, example_input=example_input, input=input_shapes)
        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model, 2)
        print("✅ Thinker language model successfully converted")

        if quantization_config is not None:
            print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config)
            print("✅ Weights compression finished")

        ov.save_model(ov_model, thinker_lang_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"✅ Thinker model conversion finished. You can find results in {output_dir}")

    if not talker_embedding_path.exists():
        print("⌛ Convert talker embedding model")
        __make_16bit_traceable(model.talker.model.get_input_embeddings())
        ov_model = ov.convert_model(
            model.talker.model.get_input_embeddings(),
            example_input=torch.ones([2, 2], dtype=torch.int64),
        )
        ov.save_model(ov_model, talker_embedding_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Talker embedding model successfully converted")

    if not talker_lang_path.exists():
        print("⌛ Convert Talker Language model")

        def forward_wrap_talker(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[tuple, BaseModelOutputWithPast]:
            """take care of image_encode, position_ids and (attention_mask = None is fine)"""
            if past_key_values is not None:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            talker_lm_input = self.thinker_to_talker_proj(inputs_embeds)

            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=talker_lm_input,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if past_key_values is not None:
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)
            logits = logits.float()
            output = (logits,) + outputs[1:]

            return output

        lang_model = model.talker
        num_pkv = lang_model.model.config.num_hidden_layers
        embedding_size = lang_model.model.config.embedding_size
        lang_model._orig_forward = lang_model.forward
        lang_model.forward = types.MethodType(forward_wrap_talker, lang_model)

        num_pkv = lang_model.model.config.num_hidden_layers
        pkv_shape = (2, lang_model.model.config.num_key_value_heads, 2, lang_model.model.config.head_dim)
        # input_embeds = torch.randn((1, 1, hidden_size))
        cache_position = torch.arange(2, 4)
        position_ids = cache_position.view(1, 1, -1).expand(3, 2, -1)

        input_embeds = torch.randn((2, 2, embedding_size))
        attention_mask = torch.ones([2, 4], dtype=torch.long)
        input_names = ["attention_mask", "position_ids"]
        output_names = ["logits"]
        past_key_values = []
        for i in range(num_pkv):
            kv = [torch.randn(pkv_shape) for _ in range(2)]
            past_key_values.append(kv)
            input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        input_names.append("inputs_embeds")
        example_input = {"inputs_embeds": input_embeds, "attention_mask": attention_mask, "position_ids": position_ids, "past_key_values": past_key_values}

        input_shapes = [
            ov.PartialShape([-1, -1]),
            ov.PartialShape([3, -1, -1]),
        ]
        input_shapes += [ov.PartialShape([-1, lang_model.model.config.num_key_value_heads, -1, lang_model.model.config.head_dim])] * 2 * num_pkv
        input_shapes += [ov.PartialShape([-1, -1, input_embeds.shape[-1]])]
        __make_16bit_traceable(lang_model)

        ov_model = ov.convert_model(lang_model, example_input=example_input, input=input_shapes)
        for input, input_name in zip(ov_model.inputs, input_names):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, output_names):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model, 1)
        print("✅ Talker language model successfully converted")

        if quantization_config is not None:
            print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config)
            print("✅ Weights compression finished")

        ov.save_model(ov_model, talker_lang_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print(f"✅ Talker model conversion finished. You can find results in {output_dir}")

    if not token2wav_dit_path.exists():
        print("⌛ Convert token2wav DIT model")

        def forward_wrap_dit_attention(
            self,
            hidden_states,  # noised input x
            position_embeddings=None,  # rotary position embedding for x
            attention_mask=None,
        ) -> torch.Tensor:
            batch_size = hidden_states.shape[0]

            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            # attention
            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads
            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            # apply rotary position embedding
            # Due to training process, only first head is applied with RoPE, will be fixed at next release
            cos, sin = position_embeddings
            query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

            attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
            attn_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
            attn_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))
            attention_weights, _ = attention_interface(
                self,
                query,
                key,
                value,
                attention_mask=attn_mask,
                is_causal=False,
            )

            # mask. e.g. inference got a batch with different target durations, mask out the padding
            attention_weights = attention_weights.reshape(batch_size, -1, self.heads * head_dim)
            attention_weights = attention_weights.to(query.dtype)

            # linear proj
            attention_output = self.to_out[0](attention_weights)
            attention_output = self.to_out[1](attention_output)

            return attention_output

        code2wav_dit = model.token2wav.code2wav_dit_model
        for block in code2wav_dit.transformer_blocks:
            block.attn.forward = types.MethodType(forward_wrap_dit_attention, block.attn)

        __make_16bit_traceable(code2wav_dit)
        ov_model = ov.convert_model(
            code2wav_dit,
            example_input={
                "hidden_states": torch.randn([1, 4, model.token2wav.code2wav_dit_model.config.mel_dim], dtype=torch.float32),
                "quantized_code": torch.ones([1, 2], dtype=torch.int64),
                "speaker_embedding": torch.randn([1, 4, model.token2wav.code2wav_dit_model.config.enc_emb_dim], dtype=torch.float32),
                "condition_vector": torch.full((1, 400, model.token2wav.code2wav_dit_model.config.mel_dim), fill_value=-11.5129, dtype=torch.float32),
                "time_step": torch.tensor(0.0051, dtype=torch.float32),
            },
        )

        ov.save_model(ov_model, token2wav_dit_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Token2wav DIT model successfully converted")

    if not token2wav_bigvgan_path.exists():
        print("⌛ Convert token2wav bigvgan model")

        def forward_wrap_bigvgan(self, mel_spectrogram):
            processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
            hidden_representation = self.conv_pre(processed_spectrogram)

            for layer_index in range(self.num_upsample_layers):
                hidden_representation = self.ups[layer_index][0](hidden_representation)
                residual_output = sum(
                    self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                    for block_index in range(self.num_residual_blocks)
                )
                residual_output = residual_output / self.num_residual_blocks
                hidden_representation = residual_output

            hidden_representation = self.activation_post(hidden_representation)
            output_waveform = self.conv_post(hidden_representation)
            audio = torch.clamp(output_waveform, min=-1.0, max=1.0)
            return audio

        code2wav_bigvgan = model.token2wav.code2wav_bigvgan_model
        code2wav_bigvgan.forward = types.MethodType(forward_wrap_bigvgan, code2wav_bigvgan)
        __make_16bit_traceable(code2wav_bigvgan)
        ov_model = ov.convert_model(
            code2wav_bigvgan,
            example_input={
                "mel_spectrogram": torch.randn([1, code2wav_bigvgan.config.mel_dim, 2], dtype=torch.float32),
            },
        )
        ov.save_model(ov_model, token2wav_bigvgan_path)
        del ov_model
        cleanup_torchscript_cache()
        del model
        gc.collect()
        print("✅ Token2wav BIGVGAN model successfully converted")
        print(f"✅ {model_id} model conversion finished. You can find results in {output_dir}")
