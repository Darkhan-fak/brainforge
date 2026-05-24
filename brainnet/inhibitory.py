"""
InhibitoryLayer: Implements lateral inhibition in neural networks.

In biological cortex, ~20% of neurons are inhibitory (GABAergic).
They suppress neighboring excitatory neurons, creating:
- Sharper feature representations (winner-take-more)
- Better noise suppression
- More stable training dynamics

This module learns which neurons to suppress and by how much.
"""

import torch
import torch.nn as nn


class InhibitoryLayer(nn.Module):
    """
    Applies learned lateral inhibition to a fraction of neurons.

    Args:
        features: Number of features in the layer
        ratio: Fraction of neurons that are inhibitory (default: 0.2)

    Example:
        >>> layer = InhibitoryLayer(256, ratio=0.2)
        >>> x = torch.randn(32, 256)
        >>> out = layer(x)  # same shape, but with inhibition applied
    """

    def __init__(self, features: int, ratio: float = 0.2):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.n_inhibitory = max(1, int(features * ratio))

        # Inhibitory mask: which neurons are inhibitory
        # (learned during training via straight-through estimator)
        self.inhibitory_weights = nn.Parameter(
            torch.randn(self.n_inhibitory, features) * 0.01
        )

        # Inhibitory strength (learnable)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute inhibitory signal
        # Each inhibitory neuron reads from all excitatory neurons
        # and produces a suppression signal
        inhib_input = x[..., :self.n_inhibitory]  # inhibitory neurons read input
        inhib_signal = torch.matmul(inhib_input, self.inhibitory_weights)

        # Apply soft suppression (not hard gating)
        suppression = torch.sigmoid(inhib_signal) * self.strength

        # Subtract inhibition from excitatory activity
        return x - suppression
