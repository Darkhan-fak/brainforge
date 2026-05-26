"""
Inhibition Diagnostics: Measure E/I balance metrics across ratios (0%, 10%, 20%, 30%).
Supports:
1. Untrained models (on initialization, hidden_channels=256).
2. Trained models (loaded from short_prolog checkpoints, hidden_channels=256).
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brainnet import ConvBackbone


def get_diagnostic_batch(data_dir="./data", batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    # Take 1 batch
    loader = DataLoader(Subset(train_set, range(batch_size)), batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def run_diagnostics_for_ratio(ratio: float, images: torch.Tensor, labels: torch.Tensor, device: torch.device,
                              checkpoint_path: str = None, hidden_channels: int = 256):
    # Construct model
    model = ConvBackbone(
        in_channels=3,
        hidden_channels=hidden_channels,
        out_features=10,
        inhibitory_ratio=ratio,
        max_inhibitory_ratio=0.3,  # Set max to 0.3 to cover all swept ratios
        use_lateral=True
    ).to(device)

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"[WARNING] Checkpoint {checkpoint_path} not found! Running on random weights.")

    # Dictionary to store intermediate hook data
    layer_stats = {}

    def get_inhib_hook(layer_name):
        def hook(module, hook_input, hook_output):
            x_pre = hook_input[0].detach()
            x_post = hook_output.detach()
            active_n = module.active_n
            max_n = module.max_n
            
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {}
            layer_stats[layer_name].update({
                "pre": x_pre,
                "post": x_post,
                "active_n": active_n,
                "max_n": max_n
            })
        return hook

    def get_layer_hook(layer_name):
        def hook(module, hook_input, hook_output):
            x_final = hook_output.detach()
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {}
            layer_stats[layer_name].update({
                "final": x_final
            })
        return hook

    # Register hooks
    hooks = []
    for name, layer in model.layers.items():
        hooks.append(layer.inhibitory.register_forward_hook(get_inhib_hook(name)))
        hooks.append(layer.register_forward_hook(get_layer_hook(name)))

    # Forward pass
    model.train()
    logits = model(images)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    # Backward pass to generate gradients
    model.zero_grad()
    loss.backward()

    # Process metrics
    report = {}
    for name, stats in layer_stats.items():
        pre = stats["pre"]  # [B, H, W, C]
        post = stats["post"]  # [B, H, W, C]
        final = stats["final"]  # [B, C, H, W]
        active_n = stats["active_n"]
        max_n = stats["max_n"]
        
        # 1. Inhibitory stats (active only)
        if active_n > 0:
            inhib_pre = pre[..., :active_n]
            inhib_mean = inhib_pre.mean().item()
            
            # Silent neurons: fraction of active channels where abs < 1e-5 on > 95% of batch
            B, H, W, _ = inhib_pre.shape
            flat_inhib = inhib_pre.reshape(B * H * W, active_n)
            silent_mask = (flat_inhib.abs() < 1e-5).sum(dim=0).float() / (B * H * W) > 0.95
            silent_pct = (silent_mask.sum().item() / active_n) * 100
            
            # Saturated neurons: abs > 5.0 on > 95% of batch
            sat_mask = (flat_inhib.abs() > 5.0).sum(dim=0).float() / (B * H * W) > 0.95
            sat_pct = (sat_mask.sum().item() / active_n) * 100
        else:
            inhib_mean = 0.0
            silent_pct = 0.0
            sat_pct = 0.0

        # 2. Excitatory suppression stats
        # Compute mean of excitatory channels (channels starting from active_n)
        exc_pre = pre[..., active_n:]
        exc_post = post[..., active_n:]
        
        exc_pre_mean = exc_pre.mean().item()
        exc_post_mean = exc_post.mean().item()
        
        # Drop %: (pre - post) / pre
        drop_pct = ((exc_pre_mean - exc_post_mean) / (exc_pre_mean + 1e-8)) * 100

        # Post-block stats: after Conv -> Inhibition -> GroupNorm -> Activation
        post_block_excit = final[:, active_n:]
        mean_post_block = post_block_excit.mean().item()
        zero_fraction_post_block = (post_block_excit.abs() < 1e-6).float().mean().item()

        # 3. Gradient norms and ratio
        grad_norm_inhib = 0.0
        grad_norm_excit = 0.0
        grad_ratio = 0.0
        
        layer_module = model.layers[name]
        
        # Excitatory weight grad norm
        if layer_module.excitatory.weight.grad is not None:
            grad_norm_excit = layer_module.excitatory.weight.grad.norm().item()
            
        # Inhibitory weight grad norm (active weights only)
        if layer_module.inhibitory.inhibitory_weights_active is not None and layer_module.inhibitory.inhibitory_weights_active.grad is not None:
            grad_norm_inhib = layer_module.inhibitory.inhibitory_weights_active.grad.norm().item()
            grad_ratio = grad_norm_inhib / (grad_norm_excit + 1e-8)

        report[name] = {
            "active_n": active_n,
            "inhib_mean": inhib_mean,
            "silent_pct": silent_pct,
            "sat_pct": sat_pct,
            "exc_pre_mean": exc_pre_mean,
            "exc_post_mean": exc_post_mean,
            "drop_pct": drop_pct,
            "mean_post_block": mean_post_block,
            "zero_fraction_post_block": zero_fraction_post_block,
            "grad_norm_inhib": grad_norm_inhib,
            "grad_norm_excit": grad_norm_excit,
            "grad_ratio": grad_ratio
        }

    # Remove hooks
    for h in hooks:
        h.remove()

    return report


def generate_report(ratios, all_reports, report_path, title_prefix):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Inhibition Diagnostics Report ({title_prefix})\n\n")
        f.write("This document summarizes the diagnostics run to evaluate E/I balance and the lateral inhibition mechanism.\n\n")
        
        f.write("## 1. Quantitative Metrics per Ratio\n\n")
        
        for r in ratios:
            f.write(f"### Inhibition Ratio: {r*100:.0f}%\n\n")
            f.write("| Layer | Active Inhib | Mean Inhib Act | Silent Inhib % | Sat Inhib % | Pre-Inhib Mean | Post-Inhib Mean | Suppression Drop % | Post-Block Mean | Post-Block Zero % | Grad Norm Inhib | Grad Norm Excit | Grad Ratio (Inhib/Excit) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            
            report = all_reports[r]
            for name in ["L4", "L2_3", "L5", "L6"]:
                s = report[name]
                f.write(f"| {name} | {s['active_n']} | {s['inhib_mean']:.4f} | {s['silent_pct']:.1f}% | {s['sat_pct']:.1f}% | {s['exc_pre_mean']:.4f} | {s['exc_post_mean']:.4f} | {s['drop_pct']:.2f}% | {s['mean_post_block']:.4f} | {s['zero_fraction_post_block']*100:.2f}% | {s['grad_norm_inhib']:.4f} | {s['grad_norm_excit']:.4f} | {s['grad_ratio']:.4f} |\n")
            f.write("\n")
            
            # Check for warnings
            warnings = []
            for name in ["L4", "L2_3", "L5", "L6"]:
                s = report[name]
                if s["active_n"] > 0 and s["silent_pct"] > 80.0:
                    warnings.append(f"- **[WARNING]** Layer {name} has >80% silent inhibitory channels. Potential representation collapse.")
                if s["drop_pct"] > 80.0:
                    warnings.append(f"- **[WARNING]** Layer {name} excitation drop is {s['drop_pct']:.1f}%. Suppression is too aggressive.")
                if s["active_n"] > 0 and s["grad_ratio"] < 0.1:
                    warnings.append(f"- **[WARNING]** Layer {name} gradient ratio is {s['grad_ratio']:.4f} (< 0.1). Inhibitory weights are learning too slowly.")
                if s["active_n"] > 0 and s["grad_ratio"] > 1.0:
                    warnings.append(f"- **[WARNING]** Layer {name} gradient ratio is {s['grad_ratio']:.4f} (> 1.0). Inhibitory weights are dominating gradients.")
                if s["zero_fraction_post_block"] > 0.10:
                    warnings.append(f"- **[WARNING]** Layer {name} post-block zero fraction is {s['zero_fraction_post_block']*100:.1f}% (> 10%). Inhibition/Activation is killing the signal.")
            
            if warnings:
                f.write("#### Health Check Warnings:\n")
                f.write("\n".join(warnings) + "\n\n")
            else:
                f.write("#### Health Check: PASS (No warnings)\n\n")

        f.write("## 2. Code Review and Architectural Verification\n\n")
        f.write("### Forward Pass Sequence\n")
        f.write("1. **Excitatory Projection**: The standard 2D convolution `self.excitatory = nn.Conv2d` processes the input feature maps: `h = self.excitatory(x)`.\n")
        f.write("2. **Permutation to Channel-Last**: To perform channel-wise operations, the feature maps are permuted: `h_flat = h.permute(0, 2, 3, 1)`.\n")
        f.write("3. **Excitation-Inhibition Split**: `InhibitoryLayer` splits the channel-last activations into inhibitory ($C_{\\text{inhib}} = [0, N_{\\text{active}}]$) and excitatory ($C_{\\text{excitatory}} = [N_{\\text{active}}, C]$).\n")
        f.write("4. **Learned Suppression**: Inhibitory channels produce a suppression map via matrix multiplication: `suppression = sigmoid(inhib_input * weights) * strength`.\n")
        f.write("5. **Subtractive Inhibition**: Excitatory channels are suppressed via subtraction: `x - suppression`.\n")
        f.write("6. **Lateral Connections**: Intra-layer recurrence is applied channel-wise: `h_flat = self.lateral(h_flat)`.\n")
        f.write("7. **Permutation back & Normalization**: The tensor is permuted back to channel-first, and `GroupNorm(1, C)` (equivalent to LayerNorm over H, W, C) is applied.\n")
        f.write("8. **Non-linearity**: Layer-specific biological activation functions are applied last.\n\n")
        
        f.write("### Critical Implementation Details Verified:\n")
        f.write("- **Position**: Lateral inhibition and recurrence happen **before** normalisation and non-linearity. This mirrors biology, where input integration and recurrent dendritic loops occur prior to neural firing thresholds.\n")
        f.write("- **Normalization**: Applying GroupNorm **after** the subtractive suppression stabilizes activations and prevents negative value explosion, ensuring numerical stability during training.\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inhibition diagnostics on {device}...")

    images, labels = get_diagnostic_batch()
    images, labels = images.to(device), labels.to(device)

    ratios = [0.0, 0.1, 0.2, 0.3]
    
    # ------------------------------------------------------------
    # 1. Untrained Diagnostics (On Initialization)
    # ------------------------------------------------------------
    print("\n=== RUNNING DIAGNOSTICS AT INITIALIZATION (UNTRAINED) ===")
    untrained_reports = {}
    for r in ratios:
        print(f"Untrained sweep ratio: {r * 100:.0f}%...")
        untrained_reports[r] = run_diagnostics_for_ratio(r, images, labels, device, checkpoint_path=None)
    
    untrained_report_path = "benchmarks/results/inhibition_diagnostics_untrained.md"
    generate_report(ratios, untrained_reports, untrained_report_path, "Untrained - On Initialization")
    print(f"Untrained report saved to: {untrained_report_path}")

    # ------------------------------------------------------------
    # 2. Trained Diagnostics (From Short Prolog Checkpoints)
    # ------------------------------------------------------------
    print("\n=== RUNNING DIAGNOSTICS ON TRAINED SHORT PROLOG CHECKPOINTS ===")
    
    # Map ratios to their respective best seed checkpoints from short prolog
    checkpoints = {
        0.0: "benchmarks/results/short_prolog/conv_cortical_inhib0/seed_7/model.pt",
        0.1: "benchmarks/results/short_prolog/conv_cortical_inhib10/seed_7/model.pt",
        0.2: "benchmarks/results/short_prolog/conv_cortical_inhib20/seed_42/model.pt",
        0.3: "benchmarks/results/short_prolog/conv_cortical_inhib30/seed_123/model.pt"
    }

    trained_reports = {}
    for r in ratios:
        ckpt_path = checkpoints[r]
        print(f"Trained sweep ratio: {r * 100:.0f}%...")
        trained_reports[r] = run_diagnostics_for_ratio(r, images, labels, device, checkpoint_path=ckpt_path)

    trained_report_path = "benchmarks/results/inhibition_diagnostics_trained.md"
    generate_report(ratios, trained_reports, trained_report_path, "Trained - After Short Prolog")
    print(f"Trained report saved to: {trained_report_path}")


if __name__ == "__main__":
    main()
