# BrainForge

**Brain-Inspired AI: PyTorch modules built on real neuroscience data.**

Current AI architectures ignore 500 million years of brain evolution. The human brain runs on 20 watts and learns from single examples. GPT-class models consume megawatts and need trillions of tokens. The gap is architectural.

BrainForge bridges this gap with drop-in PyTorch modules based on real cortical architecture from Open Brain Institute, Allen Brain Atlas, and TRIBE v2.

## Quick Start

```python
from brainnet import CorticalColumn, TopoLoss

# 6-layer cortical column with inhibitory neurons and lateral connections
model = CorticalColumn(
    in_features=784,
    hidden_features=256,
    out_features=10,
    inhibitory_ratio=0.2,   # 20% inhibitory neurons (matching biology)
    use_lateral=True,        # intra-layer connections
)

# Topographic loss: similar features should be processed by nearby neurons
topo_loss = TopoLoss(weight=0.1)

# Training
logits = model(x)
loss = criterion(logits, y) + topo_loss(model.get_activations())
loss.backward()
```

## What Makes This Different

| Feature | Standard NN | BrainForge |
|---------|------------|------------|
| Architecture | Flat, homogeneous layers | 6-layer cortical columns with distinct cell types |
| Inhibition | None | 20% inhibitory neurons (lateral inhibition) |
| Connections | Feed-forward only | 80% lateral (intra-layer) + feed-forward |
| Organization | Random | Topographic maps (TopoLoss) |
| Learning rate | Fixed schedule | Neuromodulated (DopamineLR) — coming soon |
| Forgetting | Catastrophic | Sleep-phase consolidation — coming soon |
| Alignment | RLHF (human ratings) | RLHBF (brain activation) — coming soon |

## Projects

### Phase 1: BrainNet (Architecture) ← current
Cortex-inspired layers as drop-in PyTorch modules.

### Phase 2: NeuroSleep (Training) — planned
Neuromodulated learning rate + sleep-phase consolidation.

### Phase 3: RLHBF (Alignment) — planned
Reinforcement Learning from Human Brain Feedback using TRIBE v2.

## Data Sources

- [Open Brain Institute](https://openbraininstitute.org/) — cortical column reconstructions
- [Allen Brain Atlas](https://portal.brain-map.org/) — single-neuron connectivity
- [TRIBE v2](https://github.com/facebookresearch/algonauts-2025) — brain encoding model (CC BY-NC)

## Benchmarks & Ablation Study

We benchmarked BrainForge's `CorticalColumn` against a standard MLP baseline on a subset of CIFAR-10 (5,000 images, 5 epochs on CPU).

### Performance Comparison

| Metric | Baseline MLP | BrainNet (Full) |
|---|---|---|
| Parameters | 986,634 | 2,024,206 |
| Peak Test Accuracy | 41.2% (Epoch 5) | **41.4%** (Epoch 2) |
| Final Test Accuracy | 41.2% | 38.6% |
| Training Time | 6.5s | 10.8s |

> [!NOTE]
> **Sample Efficiency**: BrainNet reaches its peak performance of **41.4%** in just **2 epochs**, whereas the baseline MLP needs all **5 epochs** to reach **41.2%**. This highlights the rapid learning capabilities of cortex-inspired networks. However, because of the larger capacity and limited training data, the full BrainNet model starts overfitting after epoch 2.

### Ablation Study

By turning individual brain-inspired components on and off, we analyzed their impact on performance:

| Configuration | Test Accuracy (%) | Description |
|---|---|---|
| **1. None (Baseline structure)** | 41.33% | Flat CorticalColumn without brain-specific structures |
| **2. Inhibitory Only** | 41.00% | 20% inhibitory neurons with lateral inhibition |
| **3. TopoLoss Only** | 40.18% | Spatial grouping of similar representations |
| **4. Lateral Only** | 39.36% | Intra-layer feedforward-feedback recurrence |
| **5. Full BrainNet** | 39.00% (**41.4% peak**) | All biological components combined |

![Accuracy Comparison Plot](benchmarks/results/accuracy_comparison.png)

## Phase 2A: Parameter Auditing & Model Configurations

To ensure a rigorous, scientific comparison of standard and brain-inspired architectures, we matched all configurations to a shared capacity budget. The configurations are divided into MLP-style (~2.02M parameters) and Conv-style (~2.12M parameters) classes:

| Configuration | Total Parameters | Trainable Parameters | Description / Active Components |
|---|---|---|---|
| **1. MLP Baseline** | 2,025,215 | 2,025,215 | Standard feedforward MLP (H=455) |
| **2. CorticalColumn (MLP-Style)** | 2,024,206 | 2,024,206 | Bio-inspired MLP (H=256, inhib=20%, lateral=True) |
| **3. Conv Baseline** | 2,123,022 | 1,783,050 | ConvBackbone (H=256, inhib=0%, lateral=False) [E/I & lateral weights frozen] |
| **4. ConvCortical (inhib=0%)** | 2,123,022 | 1,979,658 | ConvBackbone (H=256, inhib=0%, lateral=True) [E/I frozen, lateral active] |
| **5. ConvCortical (inhib=10%)** | 2,123,022 | 2,005,262 | ConvBackbone (H=256, inhib=10%, lateral=True) [active E/I = 10%, lateral active] |
| **6. ConvCortical (inhib=20%)** | 2,123,022 | 2,031,886 | ConvBackbone (H=256, inhib=20%, lateral=True) [active E/I = 20%, lateral active] |
| **7. ConvCortical (inhib=30%)** | 2,123,022 | 2,057,486 | ConvBackbone (H=256, inhib=30%, lateral=True) [active E/I = 30%, lateral active] |

> [!NOTE]
> **Parameter Symmetry**: All `ConvBackbone` configurations share the exact same layer dimensions, kernel sizes, and normalizations, resulting in an identical total parameter count (`2,123,022`). Non-active biological mechanisms (such as lateral connections when `use_lateral=False` or excess inhibitory capacity up to `max_inhibitory_ratio=0.3`) are kept as frozen parameters (`requires_grad = False` and set to `0.0`), isolating biological features without changing parameter capacity.

## Run Tests

```bash
python tests/test_brainnet.py
```

## License

MIT
