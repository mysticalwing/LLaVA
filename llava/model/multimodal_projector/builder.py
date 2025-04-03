import re

import torch.nn as nn

__all__ = [
    "IdentityMap",
    "SimpleResBlock",
    "build_vision_projector",
]


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # y = W*norm(x) + norm(x)
        x = self.pre_norm(x)
        x = x + self.proj(x)

        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    mm_hidden_size: int = getattr(config, "mm_hidden_size", None)
    hidden_size: int = getattr(config, "hidden_size", None)

    assert mm_hidden_size is not None
    assert hidden_size is not None

    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))

        modules = [
            nn.Linear(mm_hidden_size, hidden_size),
        ]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))

        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
