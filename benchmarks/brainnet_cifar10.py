"""
CIFAR-10 Rigorous Benchmark for BrainNet configurations.
Supports:
- Multi-seed (42, 123, 7)
- CIFAR-10 data augmentation + weight decay
- Parameter-symmetric models (MLP, ConvBaseline, ConvCortical)
- Detailed per-epoch metric logging
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brainnet import CorticalColumn, ConvBackbone, TopoLoss
from benchmarks.baseline_mlp import BaselineMLP


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(data_dir="./data", batch_size=128, subset_size=None):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if subset_size:
        train_set = Subset(train_set, range(subset_size))
        test_set = Subset(test_set, range(min(1000, len(test_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, len(train_set)


def train_one_epoch(model, loader, criterion, topo_loss, optimizer, device, is_conv):
    model.train()
    total_loss = 0
    total_ce = 0
    total_topo = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        if is_conv:
            outputs = model(images)
        else:
            x = images.view(images.size(0), -1)
            outputs = model(x)

        loss_ce = criterion(outputs, labels)
        loss = loss_ce
        
        loss_topo_val = 0.0
        if topo_loss is not None:
            loss_topo = topo_loss(model.get_activations())
            loss = loss + loss_topo
            loss_topo_val = loss_topo.item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_topo += loss_topo_val
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_topo / n, 100.0 * correct / total


def evaluate(model, loader, criterion, device, is_conv):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_conv:
                outputs = model(images)
            else:
                x = images.view(images.size(0), -1)
                outputs = model(x)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def build_model(model_type: str, kwargs: dict):
    if model_type == "mlp_baseline":
        return BaselineMLP(**kwargs)
    elif model_type == "cortical_column_mlp":
        return CorticalColumn(**kwargs)
    elif model_type == "conv_baseline" or model_type == "conv_cortical":
        return ConvBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def plot_curves(history, save_path, best_epoch=None):
    import matplotlib.pyplot as plt
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_loss, label='Train Loss')
    ax1.plot(epochs, val_loss, label='Val Loss')
    if best_epoch is not None and best_epoch in epochs:
        ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_acc, label='Train Acc')
    ax2.plot(epochs, val_acc, label='Val Acc')
    if best_epoch is not None and best_epoch in epochs:
        ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def log_event(message, log_file_path):
    import datetime
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def run_experiment(config_name: str, model_type: str, model_kwargs: dict,
                   epochs: int = 30, batch_size: int = 128, topo_weight: float = 0.0,
                   subset_size: None = None, save_dir: str = "benchmarks/results",
                   seeds: list = None, save_best_model: bool = True,
                   save_checkpoint_every: int = 10, plot_learning_curves: bool = True,
                   log_gpu_memory: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    central_log = os.path.join(save_dir, "run_log.txt")
    
    msg = f"============================================================\nRUNNING EXPERIMENT: {config_name}\n===================================================="
    print(msg)
    
    if seeds is None:
        seeds = [42, 123, 7]
        
    final_accuracies = []
    best_val_accuracies = []
    best_epochs = []
    peak_gpu_memories = []
    total_times = []
    
    is_conv = (model_type in ["conv_baseline", "conv_cortical"])
    topo_loss = TopoLoss(weight=topo_weight) if topo_weight > 0.0 else None

    train_loader, test_loader, train_samples = get_data_loaders(batch_size=batch_size, subset_size=subset_size)

    # We need n_params and n_trainable for summary. Let's initialize them.
    n_params = 0
    n_trainable = 0

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        seed_dir = os.path.join(save_dir, config_name, f"seed_{seed}")
        metrics_path = os.path.join(seed_dir, "metrics.json")
        
        # Per-Seed Resume Logic (Option A)
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    m = json.load(f)
                
                history_len = 0
                if "history" in m:
                    history_len = len(m["history"])
                elif "val_acc_per_epoch" in m:
                    history_len = len(m["val_acc_per_epoch"])
                
                if history_len >= epochs:
                    seed_acc = m.get("final_val_acc", m["history"][-1]["val_acc"] if "history" in m else 0.0)
                    seed_best_val = m.get("best_val_acc", seed_acc)
                    seed_best_epoch = m.get("best_epoch", epochs)
                    seed_peak_gpu = m.get("peak_gpu_memory_gb", 0.0)
                    seed_time = m.get("total_time_seconds", 0.0)
                    
                    msg_skip = f"[SKIP seed] {config_name}/seed_{seed} already completed ({history_len}/{epochs} epochs) with Val Acc {seed_acc}%."
                    print(msg_skip)
                    log_event(f"SKIP config={config_name} seed={seed} reason=already completed", central_log)
                    
                    final_accuracies.append(seed_acc)
                    best_val_accuracies.append(seed_best_val)
                    best_epochs.append(seed_best_epoch)
                    peak_gpu_memories.append(seed_peak_gpu)
                    total_times.append(seed_time)
                    
                    # Read parameters count from existing metrics if possible
                    n_params = m.get("total_params", n_params)
                    n_trainable = m.get("trainable_params", n_trainable)
                    continue
            except Exception as e:
                print(f"[WARNING] Failed to read {metrics_path}: {e}. Re-running seed.")

        # Log configuration start
        log_event(f"START config={config_name} seed={seed}", central_log)
        
        try:
            set_seed(seed)
            if log_gpu_memory and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                
            model = build_model(model_type, model_kwargs).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Model params - Total: {n_params:,}, Trainable: {n_trainable:,}")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

            metrics = {
                "config": config_name,
                "seed": seed,
                "total_params": n_params,
                "trainable_params": n_trainable,
                "epochs": epochs,
                "history": []
            }

            best_val_acc = -1.0
            best_epoch = -1
            seed_start = time.time()
            
            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                train_loss, train_ce, train_topo, train_acc = train_one_epoch(
                    model, train_loader, criterion, topo_loss, optimizer, device, is_conv
                )
                val_loss, val_acc = evaluate(model, test_loader, criterion, device, is_conv)
                wall_time = time.time() - epoch_start

                gpu_mem_gb = 0.0
                if log_gpu_memory and torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

                history_entry = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_ce": round(train_ce, 4),
                    "train_topo": round(train_topo, 4),
                    "train_acc": round(train_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 2),
                    "wall_time": round(wall_time, 2)
                }
                if log_gpu_memory and torch.cuda.is_available():
                    history_entry["gpu_memory_gb"] = round(gpu_mem_gb, 3)

                metrics["history"].append(history_entry)

                print(f"Epoch {epoch:2d}/{epochs} | "
                      f"Train Acc: {train_acc:.1f}% (CE loss {train_ce:.4f}) | "
                      f"Val Acc: {val_acc:.1f}% (loss {val_loss:.4f}) | "
                      f"Time: {wall_time:.1f}s")
                
                # Log epoch end
                log_event(f"EPOCH config={config_name} seed={seed} epoch={epoch} val_acc={val_acc:.4f} gpu_mem={gpu_mem_gb:.2f}GB time={wall_time:.1f}s", central_log)
                
                # Save best model by val accuracy (ties resolved in favor of later epoch)
                if save_best_model:
                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        os.makedirs(seed_dir, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(seed_dir, "model_best.pt"))

                # Periodic snapshot checkpoint
                if save_checkpoint_every and epoch % save_checkpoint_every == 0:
                    os.makedirs(seed_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(seed_dir, f"model_epoch_{epoch}.pt"))

                # Plot learning curves in real-time
                if plot_learning_curves:
                    try:
                        os.makedirs(seed_dir, exist_ok=True)
                        plot_curves(metrics["history"], os.path.join(seed_dir, "learning_curves.png"), best_epoch=best_epoch if best_epoch != -1 else None)
                    except Exception as pe:
                        print(f"[WARNING] Failed to plot curves: {pe}")

            seed_elapsed = time.time() - seed_start
            final_acc = metrics["history"][-1]["val_acc"]
            
            final_accuracies.append(final_acc)
            best_val_accuracies.append(round(best_val_acc, 2))
            best_epochs.append(best_epoch)
            
            peak_gpu_mem = 0.0
            if log_gpu_memory and torch.cuda.is_available():
                peak_gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
            peak_gpu_memories.append(round(peak_gpu_mem, 3))
            total_times.append(round(seed_elapsed, 1))

            metrics["total_time_seconds"] = round(seed_elapsed, 1)
            metrics["final_val_acc"] = final_acc
            metrics["best_val_acc"] = round(best_val_acc, 2)
            metrics["best_epoch"] = best_epoch
            metrics["peak_gpu_memory_gb"] = round(peak_gpu_mem, 3)

            # Save seed results
            os.makedirs(seed_dir, exist_ok=True)
            with open(os.path.join(seed_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
                
            # Save model checkpoint
            torch.save(model.state_dict(), os.path.join(seed_dir, "model.pt"))
            msg_fin = f"Seed {seed} finished! Val Acc: {final_acc:.2f}%. Saved to {seed_dir}"
            print(msg_fin)
            
            # Log configuration end
            log_event(f"DONE config={config_name} seed={seed} best_acc={best_val_acc:.4f} final_acc={final_acc:.4f}", central_log)
            
        except Exception as e:
            log_event(f"ERROR config={config_name} seed={seed}: {str(e)}", central_log)
            raise e

    # Aggregated Summary
    mean_acc = np.mean(final_accuracies)
    std_acc = np.std(final_accuracies)
    
    mean_best_val_acc = np.mean(best_val_accuracies) if best_val_accuracies else 0.0
    std_best_val_acc = np.std(best_val_accuracies) if best_val_accuracies else 0.0
    mean_best_epoch = np.mean(best_epochs) if best_epochs else 0.0
    
    summary = {
        "config": config_name,
        "model_type": model_type,
        "total_params": n_params,
        "trainable_params": n_trainable,
        "epochs": epochs,
        "train_samples": train_samples,
        "seeds": seeds,
        "final_val_accuracies": final_accuracies,
        "mean_val_acc": round(mean_acc, 4),
        "std_val_acc": round(std_acc, 4),
        "best_val_accuracies": best_val_accuracies,
        "mean_best_val_acc": round(mean_best_val_acc, 4),
        "std_best_val_acc": round(std_best_val_acc, 4),
        "best_epochs": best_epochs,
        "mean_best_epoch": round(mean_best_epoch, 2),
        "peak_gpu_memories": peak_gpu_memories,
        "mean_peak_gpu_memory_gb": round(np.mean(peak_gpu_memories), 3) if peak_gpu_memories else 0.0,
        "max_peak_gpu_memory_gb": round(np.max(peak_gpu_memories), 3) if peak_gpu_memories else 0.0,
        "total_times": total_times,
        "total_duration_seconds": round(sum(total_times), 1)
    }

    summary_path = os.path.join(save_dir, config_name, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n============================================================")
    print(f"SUMMARY FOR {config_name}")
    print(f"============================================================")
    print(f"Val Accuracies: {final_accuracies}")
    print(f"Mean ± Std Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Summary JSON saved to {summary_path}")
    print(f"============================================================")
    log_event(f"FINISHED EXPERIMENT: {config_name} | Mean: {mean_acc:.2f}% ± {std_acc:.2f}%", central_log)

    return summary


if __name__ == "__main__":
    # Test execution
    print("Testing CIFAR-10 training configuration on tiny subset...")
    run_experiment(
        config_name="test_run",
        model_type="mlp_baseline",
        model_kwargs={"in_features": 3072, "hidden": 64, "out_features": 10},
        epochs=2,
        subset_size=200
    )
