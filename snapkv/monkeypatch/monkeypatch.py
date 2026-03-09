from importlib.metadata import version
import warnings
import transformers
from snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37
from snapkv.monkeypatch.llama_hijack_4_51 import (
    prepare_inputs_for_generation_llama_4_51,
    replace_llama_attention_4_51,
)
from snapkv.monkeypatch.mistral_hijack_4_37 import mistral_flash_attn2_forward as mistral_flash_attn2_forward_4_37, prepare_inputs_for_generation_mistral as prepare_inputs_for_generation_mistral_4_37
from snapkv.monkeypatch.mixtral_hijack_4_37 import mixtral_flash_attn2_forward as mixtral_flash_attn2_forward_4_37, prepare_inputs_for_generation_mixtral as prepare_inputs_for_generation_mixtral_4_37
from snapkv.monkeypatch.qwen_hijack_4_51 import (
    prepare_inputs_for_generation_qwen,
    replace_qwen2_attention,
    replace_qwen3_attention,
)

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_4_37
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_4_37

def replace_llama_4_51():
    transformers_version = check_version()
    version_list = ['4.51']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV Llama support. SnapKV Llama support is tested with Transformers version {version_list}.")
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_4_51
    replace_llama_attention_4_51()

def replace_mistral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_4_37
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_4_37

def replace_mixtral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mixtral_4_37
    transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = mixtral_flash_attn2_forward_4_37

def replace_qwen2():
    transformers_version = check_version()
    version_list = ['4.51']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV Qwen support. SnapKV Qwen support is tested with Transformers version {version_list}.")
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen
    replace_qwen2_attention()

def replace_qwen2_5():
    replace_qwen2()

def replace_qwen3():
    transformers_version = check_version()
    version_list = ['4.51']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV Qwen support. SnapKV Qwen support is tested with Transformers version {version_list}.")
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen
    replace_qwen3_attention()
