"""
ConvCorticalColumn: A 6-layer convolutional neural network module inspired by the neocortical column.

Maintains biological features:
- Distinct activations per layer (L1-L6)
- 20% Inhibitory neurons (applied channel-wise)
- Lateral Connections (applied channel-wise)
- Layer-wise activations saved for TopoLoss
- Global Average Pooling (GAP) readout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .inhibitory import InhibitoryLayer
from .lateral import LateralConnections


class ConvCorticalLayer(nn.Module):
    """Single convolutional cortical layer with channel-wise inhibition and lateral connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_type: str = "L2/3",
        inhibitory_ratio: float = 0.2,
        use_lateral: bool = False,
        lateral_strength: float = 0.1,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Main excitatory projection
        self.excitatory = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Inhibitory neurons (applied channel-wise)
        self.inhibitory = InhibitoryLayer(
            out_channels,
            ratio=inhibitory_ratio
        )

        # Lateral connections (applied channel-wise)
        self.lateral = None
        if use_lateral:
            self.lateral = LateralConnections(
                out_channels,
                strength=lateral_strength
            )

        # GroupNorm (Group size 1 is equivalent to LayerNorm over [C, H, W])
        self.norm = nn.GroupNorm(1, out_channels)

        # Per-layer activation
        self.activation = self._get_activation(layer_type)

    def _get_activation(self, layer_type: str) -> nn.Module:
        activations = {
            "L1": nn.Softplus(beta=2.0),
            "L2/3": nn.GELU(),
            "L4": nn.ReLU(),
            "L5": nn.ELU(alpha=1.5),
            "L6": nn.SiLU(),
        }
        return activations.get(layer_type, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Excitatory projection
        h = self.excitatory(x) # [B, C_out, H, W]

        # Reshape to channel-last [B, H, W, C] to apply InhibitoryLayer and LateralConnections
        B, C, H, W = h.shape
        h_flat = h.permute(0, 2, 3, 1)

        # Apply channel-wise inhibition
        h_flat = self.inhibitory(h_flat)

        # Apply channel-wise lateral connections
        if self.lateral is not None:
            h_flat = self.lateral(h_flat)

        # Permute back to [B, C_out, H, W]
        h = h_flat.permute(0, 3, 1, 2)

        # Normalize
        h = self.norm(h)

        # Layer-specific activation
        h = self.activation(h)

        return h


class ConvCorticalColumn(nn.Module):
    """
    A 6-layer convolutional neocortical column module.
    Preserves structural connectivity: L4 (input) -> L2/3 -> L5 (output) -> L6 (feedback).
    Uses Global Average Pooling before readout.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 256,
        out_features: int = 10,
        inhibitory_ratio: float = 0.2,
        use_lateral: bool = True,
        lateral_strength: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_features = out_features

        # L1: Sparse skip connection (1x1 conv modulation)
        self.l1_skip = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.l1_gate = nn.Sigmoid()

        # Build layers
        self.layers = nn.ModuleDict()

        # L4: Input layer
        self.layers["L4"] = ConvCorticalLayer(
            in_channels, hidden_channels,
            layer_type="L4",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength
        )

        # L2/3: Processing layer
        self.layers["L2_3"] = ConvCorticalLayer(
            hidden_channels, hidden_channels,
            layer_type="L2/3",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength
        )

        # L5: Output layer
        self.layers["L5"] = ConvCorticalLayer(
            hidden_channels, hidden_channels,
            layer_type="L5",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength
        )

        # L6: Feedback layer
        self.layers["L6"] = ConvCorticalLayer(
            hidden_channels, hidden_channels,
            layer_type="L6",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=False,
            lateral_strength=lateral_strength
        )

        # Readout head
        self.readout = nn.Linear(hidden_channels, out_features)

        # Intermediate activations for TopoLoss
        self._activations: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activations = []
        B = x.size(0)

        # L1: skip connection
        l1_mod = self.l1_gate(self.l1_skip(x))

        # L4
        h = self.layers["L4"](x)
        # Reshape to [B, C] by pooling spatially to register activations in TopoLoss correctly
        # TopoLoss expects [B, features], so we average pool [B, C, H, W] -> [B, C]
        h_pool = F.adaptive_avg_pool2d(h, (1, 1)).view(B, -1)
        self._activations.append(h_pool)

        # L2/3
        h = self.layers["L2_3"](h) * l1_mod
        h_pool = F.adaptive_avg_pool2d(h, (1, 1)).view(B, -1)
        self._activations.append(h_pool)

        # L5
        h = self.layers["L5"](h)
        h_pool = F.adaptive_avg_pool2d(h, (1, 1)).view(B, -1)
        self._activations.append(h_pool)

        # L6 feedback
        h = h + 0.1 * self.layers["L6"](h)
        h_pool = F.adaptive_avg_pool2d(h, (1, 1)).view(B, -1)
        self._activations.append(h_pool)

        # GAP readout
        h_pooled = F.adaptive_avg_pool2d(h, (1, 1)).view(B, -1)
        logits = self.readout(h_pooled)

        return logits

    def get_activations(self) -> List[torch.Tensor]:
        return self._activations
