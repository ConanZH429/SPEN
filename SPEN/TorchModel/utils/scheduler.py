import torch
import math

from .config import Config

def get_scheduler(sheduler_type: str, optimizer: torch.optim.Optimizer, config: Config) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get the learning rate scheduler

    Args:
        sheduler_type (str): The type of the scheduler to get
        optimizer (torch.optim.Optimizer): The optimizer
        config (Config): The config
    
    Returns:
        torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler
    """
    if hasattr(config, "lr_min"):
        lr_min = config.lr_min
    elif hasattr(config, "lr0"):
        lr_min = config.lr0 / 10
    else:
        lr_min = 0.0001
    warmup_epochs = config.warmup_epochs if hasattr(config, 'warmup_epochs') else 0
    epochs = config.epochs
    

    if sheduler_type == "WarmupCosin":
        lambda_max = config.lr0 / config.lr0
        lambda_min = config.lr_min / config.lr0
        warmup_epoch = config.warmup_epochs
        max_epoch = config.epochs
        lambda0 = lambda cur_iter: lambda_min + (lambda_max-lambda_min) * cur_iter / (warmup_epoch-1) if cur_iter < warmup_epoch \
            else lambda_min + (lambda_max-lambda_min)*(1 + math.cos(math.pi * (cur_iter - warmup_epoch) / (max_epoch - warmup_epoch - 1))) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        return scheduler
    elif sheduler_type == "OnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=0.5,
                                                               patience=10,
                                                               cooldown=2,
                                                               min_lr=lr_min)
        return scheduler
    else:
        raise ValueError(f"Invalid scheduler type: {sheduler_type}")