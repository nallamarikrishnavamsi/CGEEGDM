"""
EarlyStoppingWithWarmup — ported faithfully from upstream EEGDM_original
(src/util.py). Original code, not modified beyond the import path.

EarlyStopping, except it doesn't watch the monitored metric for the first
`warmup` epochs. Used in original EEGDM's finetune config with
warmup=20, patience=3 (conf/finetune/base.yaml) — a much stricter stopping
policy than CGEEGDM's current plain EarlyStopping(patience=10), which
watches from epoch 1.
"""
from lightning.pytorch.callbacks import EarlyStopping


class EarlyStoppingWithWarmup(EarlyStopping):
    """
    EarlyStopping, except don't watch the first `warmup` epochs.
    """

    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            not self._check_on_train_epoch_end
            or self._should_skip_check(trainer)
            or trainer.current_epoch < self.warmup
        ):
            return
        self._run_early_stopping_check(trainer)

    def on_validation_end(self, trainer, pl_module):
        if (
            self._check_on_train_epoch_end
            or self._should_skip_check(trainer)
            or trainer.current_epoch < self.warmup
        ):
            return
        self._run_early_stopping_check(trainer)
