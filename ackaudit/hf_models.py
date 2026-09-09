"""Real architectures from transformers, built from config with random weights.

Only the graph structure matters for partitioning, so pretrained weights are
unnecessary and we skip the hub entirely. Sizes are kept small enough to
capture on CPU; scale them up with the --scale flag when running on a GPU box.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HF models need `pip install transformers`"
        ) from exc


class LlamaWrap(nn.Module):
    def __init__(self, layers: int = 4, hidden: int = 256, heads: int = 8, vocab: int = 1000):
        super().__init__()
        _require_transformers()
        from transformers import LlamaConfig, LlamaModel

        self.model = LlamaModel(
            LlamaConfig(
                hidden_size=hidden,
                intermediate_size=int(hidden * 2.6875),
                num_hidden_layers=layers,
                num_attention_heads=heads,
                num_key_value_heads=heads,
                vocab_size=vocab,
                use_cache=False,
            )
        )

    def forward(self, ids):
        return self.model(input_ids=ids).last_hidden_state.sum()


class ViTWrap(nn.Module):
    def __init__(self, layers: int = 4, hidden: int = 192, heads: int = 3, image: int = 64):
        super().__init__()
        _require_transformers()
        from transformers import ViTConfig, ViTModel

        self.model = ViTModel(
            ViTConfig(
                hidden_size=hidden,
                num_hidden_layers=layers,
                num_attention_heads=heads,
                intermediate_size=hidden * 4,
                image_size=image,
                patch_size=16,
            )
        )

    def forward(self, px):
        return self.model(pixel_values=px).last_hidden_state.sum()


class BertWrap(nn.Module):
    def __init__(self, layers: int = 4, hidden: int = 192, heads: int = 3, vocab: int = 1000):
        super().__init__()
        _require_transformers()
        from transformers import BertConfig, BertModel

        self.model = BertModel(
            BertConfig(
                hidden_size=hidden,
                num_hidden_layers=layers,
                num_attention_heads=heads,
                intermediate_size=hidden * 4,
                vocab_size=vocab,
            ),
            add_pooling_layer=False,
        )

    def forward(self, ids):
        return self.model(input_ids=ids).last_hidden_state.sum()


ModelSpec = tuple[Callable[[], nn.Module], Callable[[], tuple]]


def hf_models(scale: int = 1) -> dict[str, ModelSpec]:
    """Model zoo. `scale` multiplies depth and sequence length."""
    L = 4 * scale
    seq = 128 * scale
    return {
        "llama": (
            lambda: LlamaWrap(layers=L),
            lambda: (torch.randint(0, 1000, (2, seq)),),
        ),
        "vit": (
            lambda: ViTWrap(layers=L),
            lambda: (torch.randn(2, 3, 64, 64, requires_grad=True),),
        ),
        "bert": (
            lambda: BertWrap(layers=L),
            lambda: (torch.randint(0, 1000, (2, seq)),),
        ),
    }
