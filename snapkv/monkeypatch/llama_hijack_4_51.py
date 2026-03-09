from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils import logging

from snapkv.monkeypatch.qwen_hijack_4_51 import _get_cache_length, _get_max_cache_length
from snapkv.monkeypatch.snapkv_utils import init_snapkv

logger = logging.get_logger(__name__)


def llama_attention_forward_4_51(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    init_snapkv(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        kv_seq_len = key_states.shape[-2]
        if hasattr(self, "kv_seq_len") and self.kv_seq_len != 0:
            kv_seq_len += self.kv_seq_len
        else:
            kv_seq_len += _get_cache_length(past_key_value, self.layer_idx)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states, value_states = self.kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                self.num_key_value_groups,
            )
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        else:
            self.kv_seq_len += key_states.shape[-2]
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                "Falling back to eager attention."
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def prepare_inputs_for_generation_llama_4_51(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0

    cache_position = kwargs.get("cache_position")
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = _get_cache_length(past_key_values, 0)
            past_length = cache_length
            max_cache_length = _get_max_cache_length(past_key_values)
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

        if cache_position is not None:
            cache_position = cache_position[-input_ids.shape[1] :]

    position_ids = kwargs.get("position_ids")
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -input_ids.shape[1] :]

    if cache_position is None:
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = _get_cache_length(past_key_values, 0)
            else:
                past_length = self.model.layers[0].self_attn.kv_seq_len
        cache_position = torch.arange(
            past_length,
            past_length + input_ids.shape[1],
            device=input_ids.device,
        )

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )

    for key, value in kwargs.items():
        if key not in model_inputs and key not in {"position_ids", "cache_position"}:
            model_inputs[key] = value

    return model_inputs


def replace_llama_attention_4_51():
    LlamaAttention.forward = llama_attention_forward_4_51
