import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from snapkv.monkeypatch.monkeypatch import replace_qwen2, replace_qwen3


def get_cache_length(past_key_values):
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return past_key_values.get_seq_length(0)
        except TypeError:
            return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[-2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_name = args.model.lower()
    if "qwen3" in model_name:
        replace_qwen3()
    else:
        replace_qwen2()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=False,
    ).to(args.device)
    model.eval()

    model.config.window_size = 8
    model.config.max_capacity_prompt = 32
    model.config.kernel_size = 3
    model.config.pooling = "avgpool"

    prompt = (
        "SnapKV keeps only the most relevant KV entries for long-context decoding. "
        "Please answer with one short sentence about what SnapKV does. "
    ) * 12
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    prefill_outputs = model(
        **inputs,
        use_cache=True,
        past_key_values=DynamicCache(),
    )
    compressed_cache_len = get_cache_length(prefill_outputs.past_key_values)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    expected_final_cache_len = compressed_cache_len + args.max_new_tokens

    print(f"prompt_tokens={inputs['input_ids'].shape[-1]}")
    print(f"compressed_cache_len={compressed_cache_len}")
    print(f"expected_final_cache_len={expected_final_cache_len}")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
