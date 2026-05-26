"""
LateralConnections: Intra-layer connections for neural networks.

In biological cortex, ~80% of connections are WITHIN a layer (lateral),
not between layers (feed-forward). This is completely absent in standard
neural networks, which only have feed-forward connections.

Lateral connections enable:
- Contextual modulation (nearby neurons influence each other)
- Pattern completion (partial input → full pattern)
- Competitive dynamics (related features reinforce, unrelated suppress)
"""

import torch
import torch.nn as nn


class LateralConnections(nn.Module):
    """
    Adds learned lateral (intra-layer) connections.

    Each neuron receives input from other neurons in the SAME layer,
    weighted by a learned lateral weight matrix.

    Args:
        features: Number of features in the layer
        strength: Scaling factor for lateral signal (default: 0.1)
        iterations: Number of lateral interaction steps (default: 1)

    Example:
        >>> lateral = LateralConnections(256, strength=0.1)
        >>> x = torch.randn(32, 256)
        >>> out = lateral(x)  # same shape, enriched by lateral context
    """

    def __init__(
        self,
        features: int,
        strength: float = 0.1,
        iterations: int = 1,
    ):
        super().__init__()
        self.features = features
        self.strength = strength
        self.iterations = iterations

        # Lateral weight matrix (sparse initialization)
        self.lateral_weight = nn.Parameter(
            torch.randn(features, features) * 0.01
        )

        # Zero out diagonal (neuron doesn't connect to itself)
        with torch.no_grad():
            self.lateral_weight.fill_diagonal_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if len(orig_shape) == 4:
            B, H, W, C = orig_shape
            h = x.reshape(B * H * W, C)
        else:
            h = x

        for _ in range(self.iterations):
            # Compute lateral signal
            lateral_signal = torch.matmul(h, self.lateral_weight)

            # Apply with tanh to bound the contribution
            lateral_signal = torch.tanh(lateral_signal) * self.strength

            # Add lateral context to current activation
            h = h + lateral_signal

        if len(orig_shape) == 4:
            return h.reshape(B, H, W, C)
        else:
            return h
