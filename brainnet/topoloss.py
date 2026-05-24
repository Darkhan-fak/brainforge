"""
TopoLoss: Topographic Organization Loss for Neural Networks.

In biological cortex, neurons that process similar features are
physically close to each other (topographic maps). This organization
emerges from development and is maintained by Hebbian learning.

TopoLoss encourages similar organization in artificial networks:
neurons with similar activation patterns should have nearby indices.
This has been shown to improve efficiency by 20%+ (Georgia Tech, 2025).

Reference:
    "TopoLoss: Topology-Aware Loss for Neural Network Efficiency"
    Georgia Tech, 2025.
"""

import torch
import torch.nn as nn
from typing import List


class TopoLoss(nn.Module):
    """
    Loss function that penalizes topographic disorganization.

    Neurons with similar activations should be near each other
    in the layer's index space. Penalty increases with distance
    between similarly-activated neurons.

    Args:
        weight: Scaling factor for topographic loss (default: 0.1)
        neighborhood: Size of local neighborhood to consider (default: 5)

    Example:
        >>> topo = TopoLoss(weight=0.1)
        >>> activations = [layer1_out, layer2_out, layer3_out]
        >>> loss = topo(activations)
    """

    def __init__(self, weight: float = 0.1, neighborhood: int = 5):
        super().__init__()
        self.weight = weight
        self.neighborhood = neighborhood

    def forward(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute topographic loss over a list of layer activations.

        Args:
            activations: List of tensors [batch, features] from each layer

        Returns:
            Scalar loss value
        """
        total_loss = torch.tensor(0.0, device=activations[0].device)

        for act in activations:
            total_loss = total_loss + self._layer_topo_loss(act)

        return self.weight * total_loss / len(activations)

    def _layer_topo_loss(self, act: torch.Tensor) -> torch.Tensor:
        """
        For a single layer: penalize if similar activations
        are far apart in index space.
        """
        # act shape: [batch, features]
        batch_size, n_features = act.shape

        if n_features <= self.neighborhood:
            return torch.tensor(0.0, device=act.device)

        # Compute activation similarity (correlation across batch)
        # Normalize activations
        act_norm = act - act.mean(dim=0, keepdim=True)
        std = act_norm.std(dim=0, keepdim=True) + 1e-8
        act_norm = act_norm / std

        # Correlation matrix [features, features]
        corr = torch.matmul(act_norm.t(), act_norm) / batch_size

        # Distance matrix (how far apart in index space)
        indices = torch.arange(n_features, dtype=torch.float32, device=act.device)
        dist = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()

        # Penalty: high correlation × high distance = bad
        # We want correlated neurons to be close
        penalty = (corr.abs() * dist).mean()

        return penalty
