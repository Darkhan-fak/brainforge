"""
Short Prolog runner for Phase 2A.

Runs 7 configurations x 3 seeds x 10 epochs on full CIFAR-10
with fixed hyperparameter budget.

Supports resume: skips configurations where summary.json already exists.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.brainnet_cifar10 import run_experiment


# ========================================
# FIXED HYPERPARAMETER BUDGET (Phase 2A)
# ========================================
# Note: learning_rate (1e-3), weight_decay (5e-4), and augmentation (True)
# are hardcoded to these exact values inside benchmarks/brainnet_cifar10.py
COMMON_PARAMS = {
    'epochs': 10,
    'seeds': [42, 123, 7],
    'subset_size': None,        # full CIFAR-10
    'batch_size': 128,
    'save_dir': 'benchmarks/results/short_prolog',
}

TOPO_WEIGHT = 0.01              # for all BrainNet configs

SAVE_DIR = 'benchmarks/results/short_prolog'


# ========================================
# 7 CONFIGURATIONS
# ========================================
# All ConvBackbone configurations use max_inhibitory_ratio = 0.3
# to maintain strict parameter symmetry (total: 2,123,022 parameters).
CONFIGS = [
    {
        'config_name': 'mlp_baseline',
        'model_type': 'mlp_baseline',
        'model_kwargs': {'in_features': 3072, 'hidden': 455, 'out_features': 10},
        'topo_weight': 0.0,
    },
    {
        'config_name': 'cortical_column_mlp',
        'model_type': 'cortical_column_mlp',
        'model_kwargs': {'in_features': 3072, 'hidden_features': 256, 'out_features': 10,
                         'inhibitory_ratio': 0.2, 'use_lateral': True},
        'topo_weight': TOPO_WEIGHT,
    },
    {
        'config_name': 'conv_baseline',
        'model_type': 'conv_cortical',
        'model_kwargs': {'in_channels': 3, 'hidden_channels': 256, 'out_features': 10,
                         'inhibitory_ratio': 0.0, 'max_inhibitory_ratio': 0.3, 'use_lateral': False},
        'topo_weight': 0.0,
    },
    {
        'config_name': 'conv_cortical_inhib0',
        'model_type': 'conv_cortical',
        'model_kwargs': {'in_channels': 3, 'hidden_channels': 256, 'out_features': 10,
                         'inhibitory_ratio': 0.0, 'max_inhibitory_ratio': 0.3, 'use_lateral': True},
        'topo_weight': TOPO_WEIGHT,
    },
    {
        'config_name': 'conv_cortical_inhib10',
        'model_type': 'conv_cortical',
        'model_kwargs': {'in_channels': 3, 'hidden_channels': 256, 'out_features': 10,
                         'inhibitory_ratio': 0.1, 'max_inhibitory_ratio': 0.3, 'use_lateral': True},
        'topo_weight': TOPO_WEIGHT,
    },
    {
        'config_name': 'conv_cortical_inhib20',
        'model_type': 'conv_cortical',
        'model_kwargs': {'in_channels': 3, 'hidden_channels': 256, 'out_features': 10,
                         'inhibitory_ratio': 0.2, 'max_inhibitory_ratio': 0.3, 'use_lateral': True},
        'topo_weight': TOPO_WEIGHT,
    },
    {
        'config_name': 'conv_cortical_inhib30',
        'model_type': 'conv_cortical',
        'model_kwargs': {'in_channels': 3, 'hidden_channels': 256, 'out_features': 10,
                         'inhibitory_ratio': 0.3, 'max_inhibitory_ratio': 0.3, 'use_lateral': True},
        'topo_weight': TOPO_WEIGHT,
    },
]


# ========================================
# RESUME LOGIC + EXECUTION
# ========================================
def is_done(config_name):
    summary_path = os.path.join(SAVE_DIR, config_name, 'summary.json')
    return os.path.exists(summary_path)


def main():
    print(f"=== SHORT PROLOG ===")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Seeds:   {COMMON_PARAMS['seeds']}")
    print(f"Epochs:  {COMMON_PARAMS['epochs']}")
    print(f"Dataset: full CIFAR-10 (subset_size=None)")
    print()
    
    skipped = []
    completed = []
    failed = []
    
    for cfg in CONFIGS:
        name = cfg['config_name']
        
        if is_done(name):
            print(f"[SKIP] {name} - summary.json exists")
            skipped.append(name)
            continue
        
        print(f"\n[RUN]  {name}")
        try:
            run_experiment(
                config_name=name,
                model_type=cfg['model_type'],
                model_kwargs=cfg['model_kwargs'],
                topo_weight=cfg['topo_weight'],
                **COMMON_PARAMS,
            )
            completed.append(name)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((name, str(e)))
    
    # ========================================
    # SUMMARY TABLE
    # ========================================
    print("\n" + "=" * 60)
    print("SHORT PROLOG SUMMARY")
    print("=" * 60)
    print(f"Skipped:   {len(skipped)} ({skipped})")
    print(f"Completed: {len(completed)} ({completed})")
    print(f"Failed:    {len(failed)}")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}")
    
    # Aggregate results
    print("\n" + "-" * 60)
    print(f"{'Config':<30} {'Mean Acc':>10} {'Std':>8} {'Epoch t':>10}")
    print("-" * 60)
    for cfg in CONFIGS:
        name = cfg['config_name']
        summary_path = os.path.join(SAVE_DIR, name, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            mean_acc = s.get('mean_val_acc', 0.0)
            std_acc = s.get('std_val_acc', 0.0)
            
            # Extract epoch time from history in first seed metrics
            epoch_time_str = "TBD"
            first_seed = COMMON_PARAMS['seeds'][0]
            metrics_path = os.path.join(SAVE_DIR, name, f"seed_{first_seed}", 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path) as mf:
                    m = json.load(mf)
                    history = m.get('history', [])
                    if history:
                        times = [h.get('wall_time', 0.0) for h in history]
                        avg_time = sum(times) / len(times)
                        epoch_time_str = f"{avg_time:.1f}s"
            
            print(f"{name:<30} {mean_acc:>10.2f} {std_acc:>8.2f} {epoch_time_str:>10}")
        else:
            print(f"{name:<30} {'N/A':>10}")
    print("-" * 60)


if __name__ == '__main__':
    main()
