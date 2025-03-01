import torch
import rich

from pathlib import Path
from .callback import Callback


class Checkpoint(Callback):
    def __init__(self, 
                 dirpath: str,
                 filename: str,
                 monitor: str,
                 monitor_mode: str = "min",
                 save_last: bool = True,
                 verbose: bool = True
                 ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = Path(filename + ".pth")
        self.monitor = monitor
        self.monitor_fn = (lambda new, old: old is None or new < old) if monitor_mode == "min" else \
                            (lambda new, old: old is None or new > old)
        self.monitor_mode = monitor_mode
        self.save_last = save_last
        self.best = None
        self.verbose = verbose
    
    def on_val_epoch_end(self, trainer):
        metric = trainer.metrics_dict[self.monitor]
        filename = self.filename.name.replace("{train}", f"{trainer.now_epoch}")
        # Save best model
        if self.monitor_fn(metric, self.best):
            self.best = metric
            torch.save(trainer.model.state_dict(), self.dirpath / filename)
            if self.verbose:
                rich.print(f"Save best model with {self.monitor}: {self.best} at {self.dirpath / filename}")
            trainer.logger.log_file(str(self.dirpath / filename))
        # Save last model
        if self.save_last and trainer.now_epoch == trainer.config.epochs - 1:
            torch.save(trainer.model.state_dict(), self.dirpath / "last.pth")
            if self.verbose:
                rich.print(f"Save last model at {self.dirpath / 'last.pth'}")
            trainer.logger.log_file(str(self.dirpath / "last.pth"))