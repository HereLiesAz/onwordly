from __future__ import annotations

from typing import Any

from onwordly.models.base import TrainStepMetrics


class HuggingFaceCausalLMAdapter:
    """Minimal causal-LM adapter with exact per-example token accounting.

    Heavy dependencies are imported lazily so the core Onwordly package and tests do
    not require PyTorch or Transformers.
    """

    def __init__(
        self,
        model_name: str,
        *,
        learning_rate: float = 2e-5,
        max_new_tokens: int = 32,
        device: str = "auto",
        gradient_clip_norm: float | None = 1.0,
        seed: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Training dependencies are missing. Install with: pip install -e '.[train]'"
            ) from exc

        self.torch = torch
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.gradient_clip_norm = gradient_clip_norm
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.device_name = str(self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _training_ids(self, prompt: str, target: str) -> tuple[list[int], list[int]]:
        prompt_text = prompt.rstrip() + "\n"
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target.strip(), add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            target_ids = [*target_ids, self.tokenizer.eos_token_id]
        if not target_ids:
            raise ValueError("target produced no tokens")
        return prompt_ids, target_ids

    def count_training_tokens(self, prompt: str, target: str) -> int:
        prompt_ids, target_ids = self._training_ids(prompt, target)
        return len(prompt_ids) + len(target_ids)

    def generate(self, prompt: str) -> str:
        torch = self.torch
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if was_training:
            self.model.train()

        generated_ids = output[0, prompt_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        torch = self.torch
        prompt_ids, target_ids = self._training_ids(prompt, target)
        all_ids = prompt_ids + target_ids
        labels = ([-100] * len(prompt_ids)) + target_ids

        input_ids = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        label_tensor = torch.tensor([labels], dtype=torch.long, device=self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs: Any = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_tensor,
        )
        loss = outputs.loss
        loss.backward()
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        return TrainStepMetrics(loss=float(loss.detach().cpu()), tokens=len(all_ids))

    def synchronize(self) -> None:
        """Wait for queued accelerator work so wall-clock measurements are honest."""
        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            synchronize = getattr(torch.mps, "synchronize", None)
            if callable(synchronize):
                synchronize()

    def reset_peak_memory_stats(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_bytes(self) -> int | None:
        if self.device.type == "cuda":
            self.synchronize()
            return int(self.torch.cuda.max_memory_allocated(self.device))
        return None

    def close(self) -> None:
        """Release model/optimizer state between regimes when the backend supports it."""
        import gc

        self.synchronize()
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(self.torch, "mps"):
            empty_cache = getattr(self.torch.mps, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
