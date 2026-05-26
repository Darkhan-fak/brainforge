# Inhibition Diagnostics Report (Trained - After Short Prolog)

This document summarizes the diagnostics run to evaluate E/I balance and the lateral inhibition mechanism.

## 1. Quantitative Metrics per Ratio

### Inhibition Ratio: 0%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 0.0000 | 0.0% | 0.0% | 0.0136 | 0.0136 | 0.00% | 0.1481 | 67.68% | 0.0000 | 2.0366 | 0.0000 |
| L2_3 | 0 | 0.0000 | 0.0% | 0.0% | -0.6114 | -0.6114 | -0.00% | 0.0050 | 0.34% | 0.0000 | 1.4244 | 0.0000 |
| L5 | 0 | 0.0000 | 0.0% | 0.0% | 0.0123 | 0.0123 | 0.00% | 0.0428 | 0.00% | 0.0000 | 0.4347 | 0.0000 |
| L6 | 0 | 0.0000 | 0.0% | 0.0% | 0.0523 | 0.0523 | 0.00% | 0.0382 | 0.02% | 0.0000 | 0.0239 | 0.0000 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 post-block zero fraction is 67.7% (> 10%). Inhibition/Activation is killing the signal.

### Inhibition Ratio: 10%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 25 | 0.0137 | 0.0% | 0.0% | 0.0197 | -0.0528 | 368.17% | 0.1400 | 69.19% | 0.0230 | 2.1452 | 0.0107 |
| L2_3 | 25 | -0.5578 | 0.0% | 0.0% | -0.6073 | -0.6077 | -0.08% | 0.0063 | 0.31% | 0.0003 | 2.0266 | 0.0002 |
| L5 | 25 | 0.0137 | 0.0% | 0.0% | 0.0128 | 0.0019 | 84.90% | 0.0343 | 0.00% | 0.0009 | 0.4940 | 0.0019 |
| L6 | 25 | 0.0440 | 0.0% | 0.0% | 0.0260 | 0.0260 | 0.01% | 0.0325 | 0.03% | 0.0000 | 0.0336 | 0.0000 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 excitation drop is 368.2%. Suppression is too aggressive.
- **[WARNING]** Layer L4 gradient ratio is 0.0107 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 69.2% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0002 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 excitation drop is 84.9%. Suppression is too aggressive.
- **[WARNING]** Layer L5 gradient ratio is 0.0019 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 gradient ratio is 0.0000 (< 0.1). Inhibitory weights are learning too slowly.

### Inhibition Ratio: 20%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 51 | 0.0211 | 0.0% | 0.0% | 0.0160 | -0.0523 | 426.40% | 0.1385 | 69.28% | 0.0258 | 1.9368 | 0.0133 |
| L2_3 | 51 | -0.8465 | 0.0% | 0.0% | -0.4226 | -0.4206 | 0.47% | 0.0065 | 0.35% | 0.0014 | 1.2288 | 0.0012 |
| L5 | 51 | 0.0102 | 0.0% | 0.0% | 0.0078 | 0.0012 | 84.38% | 0.0400 | 0.00% | 0.0005 | 0.3545 | 0.0013 |
| L6 | 51 | -0.0265 | 0.0% | 0.0% | 0.0075 | 0.0075 | 0.00% | 0.0414 | 0.04% | 0.0000 | 0.0211 | 0.0000 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 excitation drop is 426.4%. Suppression is too aggressive.
- **[WARNING]** Layer L4 gradient ratio is 0.0133 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 69.3% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0012 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 excitation drop is 84.4%. Suppression is too aggressive.
- **[WARNING]** Layer L5 gradient ratio is 0.0013 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 gradient ratio is 0.0000 (< 0.1). Inhibitory weights are learning too slowly.

### Inhibition Ratio: 30%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 76 | 0.0154 | 0.0% | 0.0% | 0.0256 | -0.0660 | 358.21% | 0.1391 | 67.61% | 0.0576 | 2.6863 | 0.0214 |
| L2_3 | 76 | -0.4602 | 0.0% | 0.0% | -0.4442 | -0.4434 | 0.17% | -0.0019 | 0.26% | 0.0010 | 2.0246 | 0.0005 |
| L5 | 76 | 0.0142 | 0.0% | 0.0% | 0.0116 | 0.0055 | 52.33% | 0.0439 | 0.00% | 0.0010 | 0.4962 | 0.0021 |
| L6 | 76 | -0.0111 | 0.0% | 0.0% | 0.0107 | 0.0107 | 0.01% | 0.0416 | 0.05% | 0.0000 | 0.0389 | 0.0000 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 excitation drop is 358.2%. Suppression is too aggressive.
- **[WARNING]** Layer L4 gradient ratio is 0.0214 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 67.6% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0005 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 gradient ratio is 0.0021 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 gradient ratio is 0.0000 (< 0.1). Inhibitory weights are learning too slowly.

## 2. Code Review and Architectural Verification

### Forward Pass Sequence
1. **Excitatory Projection**: The standard 2D convolution `self.excitatory = nn.Conv2d` processes the input feature maps: `h = self.excitatory(x)`.
2. **Permutation to Channel-Last**: To perform channel-wise operations, the feature maps are permuted: `h_flat = h.permute(0, 2, 3, 1)`.
3. **Excitation-Inhibition Split**: `InhibitoryLayer` splits the channel-last activations into inhibitory ($C_{\text{inhib}} = [0, N_{\text{active}}]$) and excitatory ($C_{\text{excitatory}} = [N_{\text{active}}, C]$).
4. **Learned Suppression**: Inhibitory channels produce a suppression map via matrix multiplication: `suppression = sigmoid(inhib_input * weights) * strength`.
5. **Subtractive Inhibition**: Excitatory channels are suppressed via subtraction: `x - suppression`.
6. **Lateral Connections**: Intra-layer recurrence is applied channel-wise: `h_flat = self.lateral(h_flat)`.
7. **Permutation back & Normalization**: The tensor is permuted back to channel-first, and `GroupNorm(1, C)` (equivalent to LayerNorm over H, W, C) is applied.
8. **Non-linearity**: Layer-specific biological activation functions are applied last.

### Critical Implementation Details Verified:
- **Position**: Lateral inhibition and recurrence happen **before** normalisation and non-linearity. This mirrors biology, where input integration and recurrent dendritic loops occur prior to neural firing thresholds.
- **Normalization**: Applying GroupNorm **after** the subtractive suppression stabilizes activations and prevents negative value explosion, ensuring numerical stability during training.
