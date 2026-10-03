import json
from pathlib import Path

import pytest

from onwordly.experiments.manifest import ArithmeticExperimentManifest

LORA_MANIFEST = Path("experiments/002-adaptive-ablation/manifest-lora.json")
FULL_MANIFEST = Path("experiments/002-adaptive-ablation/manifest.json")


def test_lora_manifest_differs_only_in_method() -> None:
    full = json.loads(FULL_MANIFEST.read_text())
    lora = json.loads(LORA_MANIFEST.read_text())
    differing = {key for key in full.keys() | lora.keys() if full.get(key) != lora.get(key)}
    assert differing == {"learning_rate", "lora"}
    assert ArithmeticExperimentManifest.from_json(LORA_MANIFEST).lora["r"] == 16
    assert ArithmeticExperimentManifest.from_json(FULL_MANIFEST).lora is None


def test_lora_manifest_rejects_incomplete_config(tmp_path) -> None:
    payload = json.loads(LORA_MANIFEST.read_text())
    del payload["lora"]["target_modules"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="target_modules"):
        ArithmeticExperimentManifest.from_json(path)


def test_lora_adapter_trains_only_adapter_weights() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from onwordly.models.huggingface import HuggingFaceCausalLMAdapter

    try:
        adapter = HuggingFaceCausalLMAdapter(
            "sshleifer/tiny-gpt2",
            device="cpu",
            seed=0,
            lora={"r": 2, "alpha": 4, "target_modules": ["c_attn"]},
        )
    except OSError:
        pytest.skip("model download unavailable")
    assert 0 < adapter.trainable_parameter_count < adapter.parameter_count
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in adapter.model.named_parameters()
        if not parameter.requires_grad
    }
    step = adapter.train_example("Compute 2 + 3.", "5")
    assert step.tokens == adapter.count_training_tokens("Compute 2 + 3.", "5")
    for name, parameter in adapter.model.named_parameters():
        if name in frozen:
            assert parameter.detach().equal(frozen[name]), name
