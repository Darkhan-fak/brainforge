"""
BrainNet: Cortex-Inspired Neural Architecture Library

Brain-inspired PyTorch modules based on real neuroscience data
from Open Brain Institute, Allen Brain Atlas, and Blue Brain Project.

Quick start:
    >>> from brainnet import CorticalColumn, TopoLoss
    >>> model = CorticalColumn(in_features=784, hidden_features=256, out_features=10)
    >>> topo_loss = TopoLoss(weight=0.1)
    >>> logits = model(x)
    >>> loss = criterion(logits, y) + topo_loss(model.get_activations())
"""

from .cortical_column import CorticalColumn, CorticalLayer
from .conv_column import ConvCorticalColumn, ConvCorticalLayer
from .conv_backbone import ConvBackbone
from .inhibitory import InhibitoryLayer
from .lateral import LateralConnections
from .topoloss import TopoLoss

__version__ = "0.1.0"
__all__ = [
    "CorticalColumn",
    "CorticalLayer",
    "ConvCorticalColumn",
    "ConvCorticalLayer",
    "ConvBackbone",
    "InhibitoryLayer",
    "LateralConnections",
    "TopoLoss",
]
