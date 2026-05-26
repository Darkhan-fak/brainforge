# Inhibition Diagnostics Report (Untrained - On Initialization)

This document summarizes the diagnostics run to evaluate E/I balance and the lateral inhibition mechanism.

## 1. Quantitative Metrics per Ratio

### Inhibition Ratio: 0%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 0.0000 | 0.0% | 0.0% | 0.0011 | 0.0011 | 0.00% | 0.3776 | 50.42% | 0.0000 | 0.3015 | 0.0000 |
| L2_3 | 0 | 0.0000 | 0.0% | 0.0% | -0.0096 | -0.0096 | -0.00% | 0.2653 | 0.01% | 0.0000 | 3.3256 | 0.0000 |
| L5 | 0 | 0.0000 | 0.0% | 0.0% | -0.0174 | -0.0174 | -0.00% | 0.0427 | 0.00% | 0.0000 | 3.5748 | 0.0000 |
| L6 | 0 | 0.0000 | 0.0% | 0.0% | -0.0069 | -0.0069 | 0.00% | 0.2016 | 0.00% | 0.0000 | 0.2314 | 0.0000 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 post-block zero fraction is 50.4% (> 10%). Inhibition/Activation is killing the signal.

### Inhibition Ratio: 10%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 25 | 0.0031 | 0.0% | 0.0% | 0.0192 | -0.0308 | 260.45% | 0.3762 | 50.07% | 0.0044 | 0.3431 | 0.0128 |
| L2_3 | 25 | 0.1759 | 0.0% | 0.0% | -0.0044 | -0.0544 | -1133.23% | 0.2393 | 0.03% | 0.0073 | 3.9343 | 0.0019 |
| L5 | 25 | 0.0185 | 0.0% | 0.0% | -0.0055 | -0.0555 | -904.18% | 0.0298 | 0.00% | 0.0057 | 4.4009 | 0.0013 |
| L6 | 25 | -0.0009 | 0.0% | 0.0% | -0.0293 | -0.0793 | -170.50% | 0.1947 | 0.00% | 0.0004 | 0.2723 | 0.0015 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 excitation drop is 260.5%. Suppression is too aggressive.
- **[WARNING]** Layer L4 gradient ratio is 0.0128 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 50.1% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0019 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 gradient ratio is 0.0013 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 gradient ratio is 0.0015 (< 0.1). Inhibitory weights are learning too slowly.

### Inhibition Ratio: 20%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 51 | 0.0125 | 0.0% | 0.0% | -0.0109 | -0.0609 | -456.57% | 0.3798 | 50.34% | 0.0058 | 0.3393 | 0.0170 |
| L2_3 | 51 | 0.0124 | 0.0% | 0.0% | -0.0259 | -0.0759 | -193.05% | 0.2475 | 0.01% | 0.0074 | 3.3960 | 0.0022 |
| L5 | 51 | -0.0253 | 0.0% | 0.0% | -0.0081 | -0.0581 | -616.96% | 0.0567 | 0.00% | 0.0078 | 3.1516 | 0.0025 |
| L6 | 51 | -0.0482 | 0.0% | 0.0% | 0.0615 | 0.0115 | 81.34% | 0.2187 | 0.00% | 0.0005 | 0.2242 | 0.0021 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 gradient ratio is 0.0170 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 50.3% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0022 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 gradient ratio is 0.0025 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 excitation drop is 81.3%. Suppression is too aggressive.
- **[WARNING]** Layer L6 gradient ratio is 0.0021 (< 0.1). Inhibitory weights are learning too slowly.

### Inhibition Ratio: 30%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 76 | -0.0084 | 0.0% | 0.0% | -0.0001 | -0.0501 | -43540.99% | 0.3697 | 49.83% | 0.0079 | 0.3107 | 0.0254 |
| L2_3 | 76 | 0.0160 | 0.0% | 0.0% | -0.0101 | -0.0602 | -494.03% | 0.2407 | 0.01% | 0.0098 | 3.4631 | 0.0028 |
| L5 | 76 | 0.0268 | 0.0% | 0.0% | -0.0033 | -0.0534 | -1493.20% | -0.0058 | 0.00% | 0.0101 | 3.4490 | 0.0029 |
| L6 | 76 | -0.0588 | 0.0% | 0.0% | 0.0162 | -0.0338 | 308.75% | 0.2209 | 0.00% | 0.0006 | 0.2271 | 0.0029 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 gradient ratio is 0.0254 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 49.8% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0028 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 gradient ratio is 0.0029 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 excitation drop is 308.8%. Suppression is too aggressive.
- **[WARNING]** Layer L6 gradient ratio is 0.0029 (< 0.1). Inhibitory weights are learning too slowly.

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
