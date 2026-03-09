from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb,
    eager_attention_forward as qwen2_eager_attention_forward,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    eager_attention_forward as qwen3_eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

from snapkv.monkeypatch.snapkv_utils import init_snapkv

logger = logging.get_logger(__name__)


def _get_cache_length(past_key_value: Cache, layer_idx: int) -> int:
    try:
        return past_key_value.get_seq_length(layer_idx)
    except TypeError:
        return past_key_value.get_seq_length()


def _get_max_cache_length(past_key_value: Cache):
    if hasattr(past_key_value, "get_max_length"):
        return past_key_value.get_max_length()
    if hasattr(past_key_value, "get_max_cache_shape"):
        max_cache_shape = past_key_value.get_max_cache_shape()
        if isinstance(max_cache_shape, int):
            return max_cache_shape
        if isinstance(max_cache_shape, (tuple, list)) and len(max_cache_shape) > 0:
            return max_cache_shape[-1]
    return None


def _update_past_key_values(
    self,
    key_states: torch.Tensor,
    query_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache],
    cache_kwargs: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if past_key_value is None:
        return key_states, value_states

    kv_seq_len = key_states.shape[-2]
    if hasattr(self, "kv_seq_len") and self.kv_seq_len != 0:
        kv_seq_len += self.kv_seq_len
    else:
        kv_seq_len += _get_cache_length(past_key_value, self.layer_idx)

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

    return key_states, value_states


def _qwen_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    rotary_fn=None,
    eager_attention_fn=None,
    use_qk_norm: bool = False,
    **kwargs,
):
    init_snapkv(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    if use_qk_norm:
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = rotary_fn(query_states, key_states, cos, sin)

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = _update_past_key_values(
        self,
        key_states,
        query_states,
        value_states,
        attention_mask,
        past_key_value,
        cache_kwargs,
    )

    attention_interface = eager_attention_fn
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
        sliding_window=getattr(self, "sliding_window", None),
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    return _qwen_attention_forward(
        self,
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        cache_position=cache_position,
        rotary_fn=qwen2_apply_rotary_pos_emb,
        eager_attention_fn=qwen2_eager_attention_forward,
        use_qk_norm=False,
        **kwargs,
    )


def qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    return _qwen_attention_forward(
        self,
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        cache_position=cache_position,
        rotary_fn=qwen3_apply_rotary_pos_emb,
        eager_attention_fn=qwen3_eager_attention_forward,
        use_qk_norm=True,
        **kwargs,
    )


def prepare_inputs_for_generation_qwen(
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


def replace_qwen2_attention():
    Qwen2Attention.forward = qwen2_attention_forward


def replace_qwen3_attention():
    Qwen3Attention.forward = qwen3_attention_forward
