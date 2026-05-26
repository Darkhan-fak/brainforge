"""
InhibitoryLayer: Implements lateral inhibition in neural networks.

In biological cortex, ~20% of neurons are inhibitory (GABAergic).
They suppress neighboring excitatory neurons, creating:
- Sharper feature representations (winner-take-more)
- Better noise suppression
- More stable training dynamics

This module learns which neurons to suppress and by how much.
It supports parameter-symmetric active/inactive splitting to guarantee
identical model capacity (total parameters) across different active ratios.
"""

import torch
import torch.nn as nn
from typing import Optional


class InhibitoryLayer(nn.Module):
    """
    Applies learned lateral inhibition to a fraction of neurons.

    Args:
        features: Number of features in the layer
        ratio: Total capacity ratio of inhibitory neurons (default: 0.2)
        active_ratio: Fraction of active inhibitory neurons (defaults to ratio)

    Example:
        >>> layer = InhibitoryLayer(256, ratio=0.2, active_ratio=0.1)
        >>> x = torch.randn(32, 256)
        >>> out = layer(x)
    """

    def __init__(self, features: int, ratio: float = 0.2, active_ratio: Optional[float] = None):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.active_ratio = active_ratio if active_ratio is not None else ratio
        
        # Max capacity sizing
        self.max_n = max(1, int(features * ratio))
        
        # Active capacity sizing
        self.active_n = int(features * self.active_ratio)
        if self.active_ratio == 0.0:
            self.active_n = 0
            
        self.inactive_n = self.max_n - self.active_n

        # Active weights (trainable)
        if self.active_n > 0:
            self.inhibitory_weights_active = nn.Parameter(
                torch.randn(self.active_n, features) * 0.01
            )
        else:
            self.register_parameter('inhibitory_weights_active', None)

        # Inactive weights (frozen to zero)
        if self.inactive_n > 0:
            self.inhibitory_weights_inactive = nn.Parameter(
                torch.zeros(self.inactive_n, features),
                requires_grad=False
            )
        else:
            self.register_parameter('inhibitory_weights_inactive', None)

        # Inhibitory strength (learnable if active, otherwise frozen to zero)
        self.strength = nn.Parameter(torch.tensor(0.1))
        if self.active_ratio == 0.0:
            self.strength.data.zero_()
            self.strength.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate active and inactive weights
        weights_list = []
        if self.inhibitory_weights_active is not None:
            weights_list.append(self.inhibitory_weights_active)
        if self.inhibitory_weights_inactive is not None:
            weights_list.append(self.inhibitory_weights_inactive)

        weights = torch.cat(weights_list, dim=0)

        # Flatten input to 2D if it is 4D (convolutional activation)
        orig_shape = x.shape
        if len(orig_shape) == 4:
            B, H, W, C = orig_shape
            x_flat = x.reshape(B * H * W, C)
        else:
            x_flat = x

        # Compute inhibitory signal
        # Excitatory and inhibitory inputs are read from the first max_n features
        inhib_input = x_flat[:, :self.max_n]
        inhib_signal = torch.matmul(inhib_input, weights)

        # Apply soft suppression (not hard gating)
        suppression = torch.sigmoid(inhib_signal) * self.strength

        # Subtract inhibition from excitatory activity
        out_flat = x_flat - suppression

        if len(orig_shape) == 4:
            return out_flat.reshape(B, H, W, C)
        else:
            return out_flat
