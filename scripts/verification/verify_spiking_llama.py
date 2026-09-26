#!/usr/bin/env python3
"""CPU verification for the checkpoint-compatible Llama adapter and SwiGLU path."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers import LlamaConfig as HFLlamaConfig
from transformers import AttentionInterface
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as HFLlamaForCausalLM,
    LlamaMLP as HFLlamaMLP,
)

from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.models.spiking_llama.configuration_llama import LlamaConfig
from utils.transformers.models.spiking_llama import modeling_spiking_llama as local_llama
from utils.transformers.integrations.spiking_sdpa_attention import (
    spiking_sdpa_attention_forward,
)


def tiny_config_kwargs() -> dict[str, object]:
    return {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16,
        "attention_dropout": 0.0,
    }


def verify_swiglu_mlp() -> None:
    """Match Hugging Face SiLU gating and prove the composed SwiGLU route is used."""
    torch.manual_seed(7)
    hf_config = HFLlamaConfig(**tiny_config_kwargs())
    hf_mlp = HFLlamaMLP(hf_config).double().eval()
    values = torch.linspace(-0.9, 0.9, 48, dtype=torch.float64).reshape(2, 3, 8)
    values = torch.cat((values, -values), dim=-1)
    potential = Potential(values, PotentialBounds(-1.0, 1.0))

    for use_spiking_mlp in (False, True):
        config = LlamaConfig(
            **tiny_config_kwargs(), use_spiking_mlp=use_spiking_mlp
        )
        local_mlp = local_llama.LlamaMLP(config).double().eval()
        local_mlp.load_state_dict(hf_mlp.state_dict(), strict=True)
        with torch.no_grad():
            expected = hf_mlp(values)
            if use_spiking_mlp:
                with patch.object(
                    local_llama,
                    "swiglu_function",
                    wraps=local_llama.swiglu_function,
                ) as composed_swiglu:
                    actual = local_mlp(potential)
                composed_swiglu.assert_called_once()
            else:
                actual = local_mlp(potential)
        torch.testing.assert_close(actual.value, expected, rtol=1e-10, atol=1e-10)
        assert actual.domain.min <= actual.value.min().item()
        assert actual.domain.max >= actual.value.max().item()


def verify_checkpoint_and_full_forward() -> None:
    """Strict-load one checkpoint and compare the complete deterministic decoder."""
    torch.manual_seed(11)
    kwargs = tiny_config_kwargs()
    hf_model = HFLlamaForCausalLM(HFLlamaConfig(**kwargs, use_cache=False)).eval()
    local_model = local_llama.LlamaForCausalLM(
        LlamaConfig(**kwargs, use_cache=False)
    ).eval()
    local_model.load_state_dict(hf_model.state_dict(), strict=True)

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.no_grad():
        expected = hf_model(input_ids).logits
        actual = local_model(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-7)

    with tempfile.TemporaryDirectory() as directory:
        hf_model.save_pretrained(directory)
        config = LlamaConfig.from_pretrained(directory, use_spiking_mlp=True)
        restored = local_llama.LlamaForCausalLM.from_pretrained(
            directory, config=config
        ).eval()
        with torch.no_grad():
            restored_logits = restored(input_ids).logits
        torch.testing.assert_close(restored_logits, actual, rtol=0.0, atol=0.0)


def verify_cache_growth() -> None:
    """Preserve Hugging Face cache semantics across incremental decoding calls."""
    torch.manual_seed(13)
    config = LlamaConfig(**tiny_config_kwargs(), use_cache=True)
    model = local_llama.LlamaForCausalLM(config).eval()
    with torch.no_grad():
        first = model(torch.tensor([[1, 2, 3]]), use_cache=True)
        assert first.past_key_values.get_seq_length() == 3
        second = model(
            torch.tensor([[4]]),
            past_key_values=first.past_key_values,
            use_cache=True,
        )
    assert second.logits.shape == (1, 1, config.vocab_size)
    assert second.past_key_values.get_seq_length() == 4
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=config.eos_token_id,
        )
    assert generated.shape == (1, 5)


def verify_spiking_grouped_attention() -> None:
    """Run shared key and value heads through the temporal attention backend."""
    AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)
    config_kwargs = {**tiny_config_kwargs(), "num_hidden_layers": 1}
    config = LlamaConfig(**config_kwargs, use_cache=False)
    model = local_llama.LlamaForCausalLM(config).eval()
    model.config._attn_implementation = "spiking_sdpa"
    with torch.no_grad():
        output = model(torch.tensor([[1, 2, 3]]))
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert bool(torch.isfinite(output.logits).all())


def verify_configuration_guards() -> None:
    """Reject obsolete global ranges and unsupported converted activations."""
    for name in ("theta", "attention_theta", "tau_m"):
        try:
            LlamaConfig(**tiny_config_kwargs(), **{name: 1.0})
        except TypeError:
            pass
        else:
            raise AssertionError(f"LlamaConfig accepted obsolete field {name}")

    config = LlamaConfig(
        **tiny_config_kwargs(), hidden_act="gelu", use_spiking_mlp=True
    )
    try:
        local_llama.LlamaMLP(config)
    except ValueError as error:
        assert "hidden_act='silu'" in str(error)
    else:
        raise AssertionError("converted Llama MLP accepted a non-SiLU activation")


def main() -> None:
    verify_swiglu_mlp()
    verify_checkpoint_and_full_forward()
    verify_cache_growth()
    verify_spiking_grouped_attention()
    verify_configuration_guards()
    print("Spiking Llama verification passed.")


if __name__ == "__main__":
    main()
