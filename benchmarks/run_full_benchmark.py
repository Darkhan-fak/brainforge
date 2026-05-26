"""
Full Benchmark Run for Phase 2A.

Runs 7 configurations x 3 seeds x 50 epochs on full CIFAR-10
with real-time plotting, checkpoints, and GPU memory tracking.

Supports resume: skips configurations where summary.json already exists,
and skips individual seeds within active configurations if their metrics.json is done.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.brainnet_cifar10 import run_experiment


# ========================================
# FIXED HYPERPARAMETER BUDGET (Phase 2A)
# ========================================
COMMON_PARAMS = {
    'epochs': 50,
    'seeds': [42, 123, 7],
    'subset_size': None,        # full CIFAR-10
    'batch_size': 128,
    'save_dir': 'benchmarks/results/full_run',
    'save_best_model': True,
    'save_checkpoint_every': 10,
    'plot_learning_curves': True,
    'log_gpu_memory': True,
}

TOPO_WEIGHT = 0.01              # for all BrainNet configs

SAVE_DIR = 'benchmarks/results/full_run'


# ========================================
# 7 CONFIGURATIONS
# ========================================
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
    print(f"=== FULL BENCHMARK RUN ===")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Seeds:   {COMMON_PARAMS['seeds']}")
    print(f"Epochs:  {COMMON_PARAMS['epochs']}")
    print(f"Dataset: full CIFAR-10")
    print(f"Save Dir: {SAVE_DIR}")
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
    print("\n" + "=" * 90)
    print("=== FULL RUN COMPLETED ===")
    print("=" * 90)
    print(f"Skipped:   {len(skipped)} ({skipped})")
    print(f"Completed: {len(completed)} ({completed})")
    print(f"Failed:    {len(failed)}")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}")
    
    # Table header
    print("\n" + "-" * 110)
    print(f"{'Configuration':<28} | {'Best Val Acc':<17} | {'Final Val Acc':<17} | {'Best Epoch':<10} | {'Train Time':<10} | {'Peak GPU Mem':<12}")
    print("-" * 110)
    
    for cfg in CONFIGS:
        name = cfg['config_name']
        summary_path = os.path.join(SAVE_DIR, name, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            
            best_mean = s.get('mean_best_val_acc', 0.0)
            best_std = s.get('std_best_val_acc', 0.0)
            final_mean = s.get('mean_val_acc', 0.0)
            final_std = s.get('std_val_acc', 0.0)
            best_ep = s.get('mean_best_epoch', 0.0)
            
            # Format time
            duration = s.get('total_duration_seconds', 0.0)
            if duration > 3600:
                time_str = f"{duration / 3600:.1f} hr"
            else:
                time_str = f"{duration / 60:.0f} min"
                
            # Format peak GPU memory
            peak_gpu = s.get('max_peak_gpu_memory_gb', 0.0)
            gpu_str = f"{peak_gpu:.1f} GB" if peak_gpu > 0.0 else "N/A"
            
            best_acc_str = f"{best_mean:.2f}% ± {best_std:.2f}%"
            final_acc_str = f"{final_mean:.2f}% ± {final_std:.2f}%"
            
            print(f"{name:<28} | {best_acc_str:<17} | {final_acc_str:<17} | {best_ep:<10.1f} | {time_str:<10} | {gpu_str:<12}")
        else:
            print(f"{name:<28} | {'N/A':<17} | {'N/A':<17} | {'N/A':<10} | {'N/A':<10} | {'N/A':<12}")
    print("-" * 110)


if __name__ == '__main__':
    main()
