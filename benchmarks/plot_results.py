"""Построение графиков точности для бенчмарков и ablation study."""
import os
import json
import matplotlib.pyplot as plt

def plot_results(results_dir="benchmarks/results"):
    files = {
        "Baseline MLP": "baseline_mlp.json",
        "BrainNet (Full)": "brainnet_cifar10.json",
        "Ablation: None": "ablation_none.json",
        "Ablation: Inhibitory Only": "ablation_inhibitory.json",
        "Ablation: Lateral Only": "ablation_lateral.json",
        "Ablation: Topo Only": "ablation_topo.json",
    }
    
    plt.figure(figsize=(10, 6))
    
    for label, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            epochs = [h["epoch"] for h in data["history"]]
            acc = [h["test_acc"] for h in data["history"]]
            plt.plot(epochs, acc, marker='o', label=label, linewidth=2)
            print(f"Loaded {label}: final accuracy {acc[-1]}%")
        else:
            print(f"Warning: {filepath} not found")
            
    plt.title("CIFAR-10 Test Accuracy over Epochs (5000 samples)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    
    save_path = os.path.join(results_dir, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_results()
