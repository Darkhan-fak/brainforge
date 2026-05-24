"""Ablation study: вклад каждого brain-inspired компонента."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.brainnet_cifar10 import run_brainnet

def run_ablation(epochs=5, subset_size=5000):
    print("=" * 60)
    print("ЗАПУСК ABLATION STUDY")
    print("=" * 60)
    
    configs = [
        {
            "name": "1. None (Baseline structure)",
            "inhibitory_ratio": 0.0,
            "use_lateral": False,
            "topo_weight": 0.0,
            "save_name": "ablation_none.json"
        },
        {
            "name": "2. Inhibitory Only",
            "inhibitory_ratio": 0.2,
            "use_lateral": False,
            "topo_weight": 0.0,
            "save_name": "ablation_inhibitory.json"
        },
        {
            "name": "3. Lateral Only",
            "inhibitory_ratio": 0.0,
            "use_lateral": True,
            "topo_weight": 0.0,
            "save_name": "ablation_lateral.json"
        },
        {
            "name": "4. TopoLoss Only",
            "inhibitory_ratio": 0.0,
            "use_lateral": False,
            "topo_weight": 0.1,
            "save_name": "ablation_topo.json"
        },
        {
            "name": "5. Full BrainNet",
            "inhibitory_ratio": 0.2,
            "use_lateral": True,
            "topo_weight": 0.1,
            "save_name": "ablation_full.json"
        }
    ]
    
    results = {}
    for cfg in configs:
        print(f"\n>>> Запуск: {cfg['name']}")
        res = run_brainnet(
            epochs=epochs,
            inhibitory_ratio=cfg["inhibitory_ratio"],
            use_lateral=cfg["use_lateral"],
            topo_weight=cfg["topo_weight"],
            subset_size=subset_size,
            save_name=cfg["save_name"]
        )
        results[cfg["name"]] = res["final_test_acc"]
        
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ABLATION STUDY")
    print("=" * 60)
    print(f"{'Конфигурация':<30} {'Точность (%)':>15}")
    print("-" * 60)
    for name, acc in results.items():
        print(f"{name:<30} {acc:>14.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_ablation()
