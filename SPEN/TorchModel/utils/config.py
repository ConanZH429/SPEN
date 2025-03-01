class Config:
    def __init__(self):
        self.lr0 = 0.001
        self.weight_decay = 0.0
        self.momentum = 0.9
        self.epochs = 100
        self.train_ratio = 1.0
        self.val_ratio = 1.0

        # optional
        self.optimizer = "AdamW"
        self.scheduler = "WarmupCosin"
        self.lr_min = 0.0001