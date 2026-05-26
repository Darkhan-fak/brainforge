# Phase 2A — Key Decisions and Lessons

## Methodological decisions

1. **Variant A (frozen weights) for parameter symmetry** - chosen over 
   Variant B (matched-width) for cleaner ablation interpretation.
2. **TopoLoss λ=0.01** - calibrated from gradient norm diagnostic 
   (default 1.0 dominated CE by 23x).
3. **3 seeds (42, 123, 7)** - minimum for std reporting.
4. **Inhibition ratios 0/10/20/30%** - biological range from V1/PFC.

## Lessons learned

1. Early hypothesis "critical period from inhibition" was wrong - 
   pattern (val > train early) exists in baseline too, due to augmentation.
2. Trainable params confound (conv_baseline vs inhib0) requires 
   Phase 2A.5 controlled experiment.
3. inhib20 had not converged at epoch 50 - ranking is sensitive to 
   training budget.
4. Multi-source AI review (strategy + coder + manual graph check) 
   caught 2 wrong claims before publication.
