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
        self.best_epoch = None
        self.verbose = verbose
    
    def on_val_epoch_end(self, trainer):
        metric = trainer.metrics_dict[self.monitor]
        # Save best model
        if self.monitor_fn(metric, self.best):
            self.best = metric
            self.best_epoch = self.trainer.now_epoch
            torch.save(trainer.model.state_dict(), self.dirpath / self.filename)
            if self.verbose:
                rich.print(f"Save best model with {self.monitor}: {self.best} at {self.dirpath / self.filename}")
            trainer.logger.log_file(str(self.dirpath / self.filename))
        # Save last model
        if self.save_last and trainer.now_epoch == trainer.config.epochs:
            torch.save(trainer.model.state_dict(), self.dirpath / "last.pth")
            if self.verbose:
                rich.print(f"Save last model at {self.dirpath / 'last.pth'}")
            trainer.logger.log_file(str(self.dirpath / "last.pth"))
            best_path = self.dirpath / self.filename
            new_name = best_path.with_name(f"{self.filename.stem}-{self.best_epoch}.{self.filename.suffix}")
            best_path.rename(new_name)