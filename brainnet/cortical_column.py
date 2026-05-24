"""
CorticalColumn: A 6-layer neural network module inspired by the neocortical column.

Based on data from Open Brain Institute (Blue Brain Project).
Each layer mimics a distinct cortical layer (L1-L6) with:
- Different activation functions per layer
- Configurable inhibitory neuron ratio
- Optional lateral (intra-layer) connections
- Optional topographic loss

Reference:
    Markram et al., "Reconstruction and Simulation of Neocortical Microcircuitry"
    Cell, 2015. DOI: 10.1016/j.cell.2015.09.029
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .inhibitory import InhibitoryLayer
from .lateral import LateralConnections


class CorticalLayer(nn.Module):
    """Single cortical layer with optional inhibition and lateral connections."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: str = "L2/3",
        inhibitory_ratio: float = 0.2,
        use_lateral: bool = False,
        lateral_strength: float = 0.1,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.in_features = in_features
        self.out_features = out_features

        # Main excitatory projection
        self.excitatory = nn.Linear(in_features, out_features)

        # Inhibitory neurons (20% by default, matching biology)
        self.inhibitory = InhibitoryLayer(
            out_features,
            ratio=inhibitory_ratio
        )

        # Lateral connections (within-layer)
        self.lateral = None
        if use_lateral:
            self.lateral = LateralConnections(
                out_features,
                strength=lateral_strength
            )

        # Layer normalization (biological analog: homeostatic plasticity)
        self.norm = nn.LayerNorm(out_features)

        # Per-layer activation (different layers use different functions)
        self.activation = self._get_activation(layer_type)

    def _get_activation(self, layer_type: str) -> nn.Module:
        """
        Different cortical layers have different response profiles.
        L4 (input): sharp, selective (ReLU)
        L2/3 (processing): graded, smooth (GELU)
        L5 (output): strong, bursting (ELU with alpha)
        L6 (feedback): moderate, sustained (SiLU/Swish)
        L1 (sparse): very sparse (Softplus with threshold)
        """
        activations = {
            "L1": nn.Softplus(beta=2.0),      # sparse, modulatory
            "L2/3": nn.GELU(),                  # smooth, graded
            "L4": nn.ReLU(),                    # sharp, selective
            "L5": nn.ELU(alpha=1.5),            # strong, bursting
            "L6": nn.SiLU(),                    # moderate, sustained
        }
        return activations.get(layer_type, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Excitatory projection
        h = self.excitatory(x)

        # Apply inhibition (20% of neurons suppress neighbors)
        h = self.inhibitory(h)

        # Lateral connections (intra-layer recurrence)
        if self.lateral is not None:
            h = self.lateral(h)

        # Normalize (homeostatic plasticity)
        h = self.norm(h)

        # Layer-specific activation
        h = self.activation(h)

        return h


class CorticalColumn(nn.Module):
    """
    A 6-layer neural network module inspired by the neocortical column.

    Architecture (following real cortical layers):
        L4 (Input)    → receives external input (like thalamic projections)
        L2/3          → intra-cortical processing
        L5 (Output)   → primary output (like pyramidal tract neurons)
        L6 (Feedback) → feedback projections
        L1 (Sparse)   → top-down modulation

    The flow is: Input → L4 → L2/3 → L5 → L6 → Output
    with L1 providing skip connections from input to L2/3.

    Args:
        in_features: Input dimension
        hidden_features: Hidden layer dimension (default: 256)
        out_features: Output dimension (e.g., number of classes)
        inhibitory_ratio: Fraction of inhibitory neurons (default: 0.2)
        use_lateral: Enable lateral connections (default: True)
        lateral_strength: Strength of lateral connections (default: 0.1)

    Example:
        >>> model = CorticalColumn(784, 256, 10)
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 10])
    """

    # Cortical layer order (biological information flow)
    LAYER_ORDER = ["L4", "L2/3", "L5", "L6"]

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        out_features: int = 10,
        inhibitory_ratio: float = 0.2,
        use_lateral: bool = True,
        lateral_strength: float = 0.1,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # L1: Sparse skip connection (input → L2/3 modulation)
        self.l1_skip = nn.Linear(in_features, hidden_features)
        self.l1_gate = nn.Sigmoid()  # gating mechanism

        # Build cortical layers
        self.layers = nn.ModuleDict()

        # L4: Input layer (receives external input)
        self.layers["L4"] = CorticalLayer(
            in_features, hidden_features,
            layer_type="L4",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength,
        )

        # L2/3: Processing layer
        self.layers["L2_3"] = CorticalLayer(
            hidden_features, hidden_features,
            layer_type="L2/3",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength,
        )

        # L5: Output layer (primary output)
        self.layers["L5"] = CorticalLayer(
            hidden_features, hidden_features,
            layer_type="L5",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=use_lateral,
            lateral_strength=lateral_strength,
        )

        # L6: Feedback layer
        self.layers["L6"] = CorticalLayer(
            hidden_features, hidden_features,
            layer_type="L6",
            inhibitory_ratio=inhibitory_ratio,
            use_lateral=False,  # L6 has fewer lateral connections
            lateral_strength=lateral_strength,
        )

        # Readout head
        self.readout = nn.Linear(hidden_features, out_features)

        # Store intermediate activations for TopoLoss
        self._activations: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._activations = []

        # L1: sparse skip connection (top-down modulation)
        l1_mod = self.l1_gate(self.l1_skip(x))

        # L4: Input processing
        h = self.layers["L4"](x)
        self._activations.append(h)

        # L2/3: Intra-cortical processing + L1 modulation
        h = self.layers["L2_3"](h) * l1_mod  # gated by L1
        self._activations.append(h)

        # L5: Primary output
        h = self.layers["L5"](h)
        self._activations.append(h)

        # L6: Feedback (residual connection to represent feedback loops)
        h = h + 0.1 * self.layers["L6"](h)
        self._activations.append(h)

        # Readout
        logits = self.readout(h)

        return logits

    def get_activations(self) -> List[torch.Tensor]:
        """Return intermediate activations for TopoLoss computation."""
        return self._activations
