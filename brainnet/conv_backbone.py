"""
ConvBackbone: A unified convolutional backbone module inspired by the neocortical column.

Biological Rationale:
---------------------
In the biological brain, mechanisms like excitation-inhibition (E/I) balance,
lateral connections (recurrent intra-layer connections), and topographic
organization (spatial mapping where adjacent neurons process similar features)
are not distinct architectures. Rather, they are superimposed biological
mechanisms built on top of the same underlying cortical substrate.

Our ConvBackbone mirrors this reality: it defines a unified neural substrate
where biological mechanisms can be dynamically engaged or disengaged through
configuration flags, without changing the network's structure.
Setting `inhibitory_ratio = 0.0` or `lateral = False` does not create a "different"
network; it simply represents a cortical column where those specific mechanisms
are disabled.

This unified architecture guarantees absolute parameter symmetry:
- The total parameter count remains constant across all configurations.
- For disabled features, the weights are frozen to zero with gradients disabled,
  meaning only the trainable parameter count changes.
- This ensures that comparative evaluations represent a true ablation of biological
  mechanisms at identical capacity, eliminating confounding architectural variables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .inhibitory import InhibitoryLayer
from .lateral import LateralConnections


class ConvLayer(nn.Module):
    """
    Unified cortical layer with optional channel-wise inhibition and lateral recurrence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_type: str = "L2/3",
        inhibitory_ratio: float = 0.0,
        max_inhibitory_ratio: float = 0.2,
        use_lateral: bool = False,
        lateral_strength: float = 0.1,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inhibitory_ratio = inhibitory_ratio
        self.use_lateral = use_lateral

        # Excitatory projection
        self.excitatory = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Inhibitory layer: sized for max_inhibitory_ratio, activated according to inhibitory_ratio
        self.inhibitory = InhibitoryLayer(
            out_channels,
            ratio=max_inhibitory_ratio,
            active_ratio=inhibitory_ratio
        )

        # Recurrent lateral connections
        # Always instantiated for param symmetry, but frozen if lateral is False
        self.lateral = LateralConnections(out_channels, strength=lateral_strength)
        if not use_lateral:
            self.lateral.lateral_weight.data.zero_()
            self.lateral.lateral_weight.requires_grad = False

        # GroupNorm (Group size 1 is equivalent to LayerNorm over [C, H, W])
        self.norm = nn.GroupNorm(1, out_channels)

        # Layer-specific activation function
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
        # Excitatory feedforward projection
        h = self.excitatory(x) # [B, C_out, H, W]

        B, C, H, W = h.shape
        # Permute to channel-last for channel-wise operations
        h_flat = h.permute(0, 2, 3, 1)

        # Apply inhibition
        h_flat = self.inhibitory(h_flat)

        # Apply recurrent lateral connections
        h_flat = self.lateral(h_flat)

        # Permute back to channel-first
        h = h_flat.permute(0, 3, 1, 2)

        # Normalize and activate
        h = self.norm(h)
        h = self.activation(h)

        return h


class ConvBackbone(nn.Module):
    """
    Unified 6-layer neocortical column backbone.
    Preserves L4 (input) -> L2/3 -> L5 (output) -> L6 (feedback) connectivity.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 256,
        out_features: int = 10,
        inhibitory_ratio: float = 0.0,
        max_inhibitory_ratio: float = 0.2,
        use_lateral: bool = False,
        lateral_strength: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_features = out_features

        # L1 skip connection (always present)
        self.l1_skip = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.l1_gate = nn.Sigmoid()

        # Neocortical layers
        self.layers = nn.ModuleDict()

        # L4 Input Layer
        self.layers["L4"] = ConvLayer(
            in_channels, hidden_channels,
            layer_type="L4",
            inhibitory_ratio=inhibitory_ratio,
            max_inhibitory_ratio=max_inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength
        )

        # L2/3 Processing Layer
        self.layers["L2_3"] = ConvLayer(
            hidden_channels, hidden_channels,
            layer_type="L2/3",
            inhibitory_ratio=inhibitory_ratio,
            max_inhibitory_ratio=max_inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength
        )

        # L5 Output Layer
        self.layers["L5"] = ConvLayer(
            hidden_channels, hidden_channels,
            layer_type="L5",
            inhibitory_ratio=inhibitory_ratio,
            max_inhibitory_ratio=max_inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength
        )

        # L6 Feedback Layer (lateral recurrent connections not used in L6 feedback layer biologically)
        self.layers["L6"] = ConvLayer(
            hidden_channels, hidden_channels,
            layer_type="L6",
            inhibitory_ratio=inhibitory_ratio,
            max_inhibitory_ratio=max_inhibitory_ratio,
            use_lateral=False,
            lateral_strength=lateral_strength
        )

        # Readout head
        self.readout = nn.Linear(hidden_channels, out_features)

        # Activation storage for TopoLoss
        self._activations: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activations = []
        B = x.size(0)

        # L1 sparse skip connection
        l1_mod = self.l1_gate(self.l1_skip(x))

        # L4
        h = self.layers["L4"](x)
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

        # L6
        h = h + 0.1 * self.layers["L6"](h)
        h_pool = F.adaptive_avg_pool2d(h, (1, 1)).view(B, -1)
        self._activations.append(h_pool)

        # Global Average Pooling (GAP) read-out
        h_pooled = F.adaptive_avg_pool2d(h, (1, 1)).view(B, -1)
        logits = self.readout(h_pooled)

        return logits

    def get_activations(self) -> List[torch.Tensor]:
        return self._activations
