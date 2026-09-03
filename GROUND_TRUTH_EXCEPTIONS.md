# Ground-Truth Exceptions

Files listed in verify_ground_truth.sh are supposed to stay byte-identical to
EEGDM_original. This file documents the ONLY approved deviations. Any drift
not listed here is unintentional and must be reverted or reviewed.

## model/diffusion_model_pl.py

### 1. sync_dist=True + on_step=True on train/val loss logging
- Lines: training_step / validation_step self.log() calls
- Reason: pretrain_hms.py runs strategy='ddp' when --devices > 1. Without
  sync_dist=True, logged loss is per-rank only and not averaged across GPUs,
  giving misleading loss curves under multi-GPU training.
- Scope: logging only. Does not change the loss computation, gradients, or
  model weights.

### 2. on_validation_epoch_end() replaced with no-op
- Original behavior: ran full reverse-diffusion sampling (up to 50 timesteps
  x n_sample x n_class), wrote .fif waveforms via mne.io.RawArray, saved
  .png plots, computed KL divergence against a target distribution.
- Reason: HMS pretraining is time-boxed to ~3 days on H100. Running full
  reverse-diffusion sampling every validation epoch is expensive and was cut
  for wall-clock budget. Checkpoint selection uses val/mse_loss (ModelCheckpoint
  monitor), which is unaffected by this change.
- Trade-off: no qualitative generated-sample artifacts (.fif/.png) are
  produced during HMS pretraining. If you need to sanity-check generation
  quality later, temporarily restore the original method from
  EEGDM_original/model/diffusion_model_pl.py, run one validation pass on a
  checkpoint, then discard the restored copy.
- Date: 2026-08-07 (per verify_ground_truth.sh diff timestamp)

## model/classifier_pl.py
- Status: intentionally absent from CGEEGDM_Final (restored 2026-09-02 after
  being found missing during ground-truth verification).
- Only consumer in the original codebase is src/finetune.py (PLClassifier),
  which finetune_graphcond.py does not use — it imports model/classifier.py
  (Classifier) instead. Kept present for original-EEGDM baseline
  reproducibility, not because CGEEGDM's active pipeline needs it.

## dataloader/TUEVDataset.py
- Kept solely as a reference copy for verify_ground_truth.sh (proves the
  original TUEV loader is unmodified). NOT used by any HMS/CGEEGDM training
  script — ConnectivityTUEVDataset.py (HMS-based) is what's actually used.
