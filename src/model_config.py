from dataclasses import dataclass
from typing import Any

@dataclass
class StableDiffusionConfig:
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    max_sequence_length: int
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_huggingface_config(cls, unet_config: Any, text_encoder_config: Any) -> "StableDiffusionConfig":
        def get_val(cfg, key, default=None):
            # Try dict-like access
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            # Try attribute access
            if hasattr(cfg, key):
                return getattr(cfg, key)
            return default

        return cls(
            vocab_size=get_val(text_encoder_config, "vocab_size", 49408),
            hidden_size=get_val(unet_config, "hidden_size", 2048),
            num_attention_heads=get_val(unet_config, "num_attention_heads", 16),
            num_hidden_layers=get_val(unet_config, "num_hidden_layers", 32),
            max_sequence_length=get_val(text_encoder_config, "max_position_embeddings", 77),
            intermediate_size=get_val(unet_config, "intermediate_size", 11008),
            rms_norm_eps=get_val(unet_config, "rms_norm_eps", 1e-6),
        )