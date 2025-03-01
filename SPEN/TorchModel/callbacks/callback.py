class Callback:
    def __init__(self):
        pass

    def on_fit_start(self, trainer):
        """Called at the beginning of the fit"""
    
    def on_train_epoch_start(self, trainer):
        """Called at the beginning of the train epoch"""

    def on_train_epoch_end(self, trainer):
        """Called at the end of the train epoch"""

    def on_val_epoch_end(self, trainer):
        """Called at the end of the validation epoch"""