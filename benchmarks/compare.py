"""Сравнение Baseline MLP vs BrainNet CorticalColumn."""
import json
import os

def compare(results_dir="benchmarks/results"):
    # Загрузить результаты
    with open(f"{results_dir}/baseline_mlp.json") as f:
        baseline = json.load(f)
    with open(f"{results_dir}/brainnet_cifar10.json") as f:
        brainnet = json.load(f)
    
    print("=" * 60)
    print("СРАВНЕНИЕ: Baseline MLP vs BrainNet CorticalColumn")
    print("=" * 60)
    print(f"{'Метрика':<30} {'Baseline':>12} {'BrainNet':>12}")
    print("-" * 60)
    print(f"{'Параметры':<30} {baseline['params']:>12,} {brainnet['params']:>12,}")
    print(f"{'Финальная точность (%)':<30} {baseline['final_test_acc']:>12.1f} {brainnet['final_test_acc']:>12.1f}")
    print(f"{'Время (сек)':<30} {baseline['total_time_seconds']:>12.1f} {brainnet['total_time_seconds']:>12.1f}")
    
    diff = brainnet['final_test_acc'] - baseline['final_test_acc']
    print(f"\n{'Разница':<30} {diff:>+12.1f}%")
    
    if diff > 0:
        print(">>> BrainNet ЛУЧШЕ baseline")
    elif diff < 0:
        print(">>> BrainNet ХУЖЕ baseline — нужна отладка")
    else:
        print(">>> Одинаково")
    
    # Сравнение по эпохам (sample efficiency)
    # На какой эпохе каждая модель достигла определённой точности?
    target_acc = 35.0  # порог для сравнения
    baseline_epoch = None
    brainnet_epoch = None
    
    for h in baseline["history"]:
        if h["test_acc"] >= target_acc and baseline_epoch is None:
            baseline_epoch = h["epoch"]
    for h in brainnet["history"]:
        if h["test_acc"] >= target_acc and brainnet_epoch is None:
            brainnet_epoch = h["epoch"]
    
    print(f"\nЭпоха достижения {target_acc}% точности:")
    print(f"  Baseline: эпоха {baseline_epoch or 'не достигнуто'}")
    print(f"  BrainNet: эпоха {brainnet_epoch or 'не достигнуто'}")
    
    if baseline_epoch and brainnet_epoch and brainnet_epoch < baseline_epoch:
        efficiency = (1 - brainnet_epoch / baseline_epoch) * 100
        print(f"  >>> BrainNet достиг цели на {efficiency:.0f}% быстрее")

if __name__ == "__main__":
    compare()
