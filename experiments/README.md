# CGEEGDM Experiments

Backbone:
- Original EEGDM backbone (unchanged)
- Pretrained backbone.ckpt

Graph branch:
- iCOH connectivity graph
- Graph encoder
- Graph-conditioned FiLM modulation
- Cosine alignment loss

Experiments:
1. Baseline
   use_graph=0
   lambda_align=0.0

2. GraphCond (No Alignment)
   use_graph=1
   lambda_align=0.0

3. GraphCond (Full)
   use_graph=1
   lambda_align=0.1

Training:
- Dataset: full106k
- Batch size: 32
- Epochs: 30
- LR: 1e-4
- Max LR: 5e-4
- Weight decay: 0.05
