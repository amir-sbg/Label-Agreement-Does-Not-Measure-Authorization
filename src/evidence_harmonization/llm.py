from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_seed(case_id: str, run_seed: int) -> int:
    digest = hashlib.sha256(f"{case_id}:{run_seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class HFTextGenerator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        sample: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.sample = sample
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        load = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, **load)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map={"": 0}, **load
            )
        except (ValueError, OSError, KeyError):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.processor = AutoTokenizer.from_pretrained(model_path, **load)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map={"": 0}, **load
            )
        self.model.eval()

    def reset_counters(self) -> None:
        self.input_tokens_total = 0
        self.output_tokens_total = 0

    def __call__(self, prompt: str, seed: int) -> str:
        self.torch.manual_seed(seed)
        random.seed(seed)
        text_only = not hasattr(self.processor, "tokenizer")
        content = prompt if text_only else [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if text_only:
            template_kwargs["enable_thinking"] = False
        try:
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items() if hasattr(value, "to")}
        n_input = inputs["input_ids"].shape[-1]
        self.input_tokens_total += int(n_input)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens, "do_sample": self.sample,
            "pad_token_id": (
                self.processor.eos_token_id if text_only
                else self.processor.tokenizer.eos_token_id
            ),
        }
        if self.sample:
            kwargs.update({"temperature": 0.7, "top_p": 0.9})
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        self.output_tokens_total += int(output.shape[-1] - n_input)
        return self.processor.batch_decode(output[:, n_input:], skip_special_tokens=True)[0].strip()
