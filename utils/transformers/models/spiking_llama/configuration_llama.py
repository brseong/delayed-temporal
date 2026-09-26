"""Checkpoint-compatible Llama configuration with TTFS conversion controls."""

from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig


class LlamaConfig(HFLlamaConfig):
    """Extend the Hugging Face Llama config with local operator controls."""

    def __init__(
        self,
        *args,
        tau_s: float = 1.0,
        use_spiking_mlp: bool = True,
        rmsnorm_clip_margin: float = 1.0e-8,
        **kwargs,
    ) -> None:
        if "theta" in kwargs or "attention_theta" in kwargs:
            raise TypeError(
                "LlamaConfig no longer accepts global or attention theta; "
                "operators use declared ranges"
            )
        if "tau_m" in kwargs:
            raise TypeError(
                "LlamaConfig accepts only tau_s; attention derives its unified tau from it"
            )
        super().__init__(*args, **kwargs)
        self.tau_s = tau_s
        self.use_spiking_mlp = use_spiking_mlp
        self.rmsnorm_clip_margin = rmsnorm_clip_margin


__all__ = ["LlamaConfig"]
