# Inhibition Diagnostics Report (Phase 2A)

This document summarizes the diagnostics run to evaluate E/I balance and the lateral inhibition mechanism.

## 1. Quantitative Metrics per Ratio

### Inhibition Ratio: 0%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 0.0000 | 0.0% | 0.0% | 0.0050 | 0.0050 | 0.00% | 0.3773 | 50.12% | 0.0000 | 0.3660 | 0.0000 |
| L2_3 | 0 | 0.0000 | 0.0% | 0.0% | -0.0554 | -0.0554 | -0.00% | 0.2588 | 0.01% | 0.0000 | 1.8621 | 0.0000 |
| L5 | 0 | 0.0000 | 0.0% | 0.0% | -0.0203 | -0.0203 | -0.00% | 0.0461 | 0.00% | 0.0000 | 1.9871 | 0.0000 |
| L6 | 0 | 0.0000 | 0.0% | 0.0% | 0.0503 | 0.0503 | 0.00% | 0.2002 | 0.00% | 0.0000 | 0.1501 | 0.0000 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 post-block zero fraction is 50.1% (> 10%). Inhibition/Activation is killing the signal.

### Inhibition Ratio: 10%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 6 | -0.1096 | 0.0% | 0.0% | -0.0114 | -0.0614 | -437.03% | 0.3993 | 49.04% | 0.0011 | 0.2680 | 0.0041 |
| L2_3 | 6 | -0.1444 | 0.0% | 0.0% | -0.0191 | -0.0691 | -260.99% | 0.2715 | 0.01% | 0.0023 | 1.5641 | 0.0015 |
| L5 | 6 | -0.0416 | 0.0% | 0.0% | 0.0279 | -0.0221 | 179.28% | 0.0730 | 0.00% | 0.0021 | 1.6091 | 0.0013 |
| L6 | 6 | -0.1038 | 0.0% | 0.0% | 0.0107 | -0.0393 | 468.85% | 0.2126 | 0.00% | 0.0001 | 0.1021 | 0.0012 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 gradient ratio is 0.0041 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 49.0% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0015 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 excitation drop is 179.3%. Suppression is too aggressive.
- **[WARNING]** Layer L5 gradient ratio is 0.0013 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 excitation drop is 468.9%. Suppression is too aggressive.
- **[WARNING]** Layer L6 gradient ratio is 0.0012 (< 0.1). Inhibitory weights are learning too slowly.

### Inhibition Ratio: 20%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 12 | 0.0216 | 0.0% | 0.0% | -0.0126 | -0.0626 | -395.44% | 0.3850 | 50.42% | 0.0026 | 0.2719 | 0.0095 |
| L2_3 | 12 | -0.0945 | 0.0% | 0.0% | -0.0153 | -0.0652 | -327.47% | 0.2794 | 0.02% | 0.0032 | 1.7833 | 0.0018 |
| L5 | 12 | 0.0329 | 0.0% | 0.0% | -0.0152 | -0.0652 | -328.85% | 0.0100 | 0.00% | 0.0040 | 1.7400 | 0.0023 |
| L6 | 12 | -0.1118 | 0.0% | 0.0% | -0.0666 | -0.1166 | -75.10% | 0.2094 | 0.00% | 0.0002 | 0.1006 | 0.0021 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 gradient ratio is 0.0095 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 50.4% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0018 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 gradient ratio is 0.0023 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 gradient ratio is 0.0021 (< 0.1). Inhibitory weights are learning too slowly.

### Inhibition Ratio: 30%

| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | 19 | 0.0028 | 0.0% | 0.0% | -0.0103 | -0.0603 | -483.90% | 0.3810 | 50.17% | 0.0033 | 0.3036 | 0.0110 |
| L2_3 | 19 | -0.0959 | 0.0% | 0.0% | -0.0295 | -0.0795 | -169.37% | 0.3019 | 0.00% | 0.0039 | 1.5846 | 0.0025 |
| L5 | 19 | -0.0275 | 0.0% | 0.0% | -0.0228 | -0.0728 | -219.21% | 0.0599 | 0.00% | 0.0047 | 2.0410 | 0.0023 |
| L6 | 19 | 0.0832 | 0.0% | 0.0% | -0.0142 | -0.0642 | -352.19% | 0.1535 | 0.00% | 0.0004 | 0.1371 | 0.0030 |

#### Health Check Warnings:
- **[WARNING]** Layer L4 gradient ratio is 0.0110 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L4 post-block zero fraction is 50.2% (> 10%). Inhibition/Activation is killing the signal.
- **[WARNING]** Layer L2_3 gradient ratio is 0.0025 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L5 gradient ratio is 0.0023 (< 0.1). Inhibitory weights are learning too slowly.
- **[WARNING]** Layer L6 gradient ratio is 0.0030 (< 0.1). Inhibitory weights are learning too slowly.

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
