import rich
import rich.table
import torch
import math
from torch import Tensor
from pathlib import Path

from ..TorchModel import Model

from ..model import SPEN
from ..cfg import SPEEDConfig, SPARKConfig
from ..utils import PosLossFunc, OriLossFunc
from ..utils import PosLoss, OriLoss, Loss, PosError, OriError, Score
from ..pose import get_ori_decoder, get_pos_decoder
from ..pose import DiscreteEuler2Euler, DiscreteSpher2Spher

from typing import Dict, Union



class ImageModule(Model):
    def __init__(self, config: Union[SPEEDConfig, SPARKConfig]):
        super().__init__()
        self.config = config
        self.BETA = self.config.BETA         # loss function weight
        self.model = SPEN(self.config)
        self.pos_decoder = get_pos_decoder(config.pos_type, **config.pos_args[config.pos_type])
        self.ori_decoder = get_ori_decoder(config.ori_type, **config.ori_args[config.ori_type])
        if self.config.pos_type == "DiscreteSpher":
            self.discrete_spher2spher = DiscreteSpher2Spher(**config.pos_args[config.pos_type])
            self.beta_0_list = [self.BETA_func(self.BETA[0], i, config.beta_epochs, 0.0) for i in range(self.config.epochs)]
        if self.config.ori_type == "DiscreteEuler":
            self.discrete_euler2euler = DiscreteEuler2Euler(**config.ori_args[config.ori_type])
            self.beta_1_list = [self.BETA_func(self.BETA[1], i, config.beta_epochs, 0.0) for i in range(self.config.epochs)]

        self._loss_init(config)

        self._metrics_init(config)
    

    def BETA_func(self, BETA, cur_iter, max_iter, min_ratio):
        if not self.config.beta_cos:
            return BETA
        if self.config.pos_type != "DiscreteSpher" or self.config.ori_type != "DiscreteEuler":
            return BETA
        if cur_iter < max_iter:
            ratio = min_ratio + (1-min_ratio)*(1 + math.cos(math.pi * cur_iter / (max_iter  - 1))) / 2
        else:
            ratio = 0
        return ratio * BETA


    def on_fit_start(self):
        table = rich.table.Table(title="hyperparameters", show_lines=True)
        dict2str = lambda d: "\n".join([f"{k} = {v}" for k, v in d.items()])
        table.add_column("hyperparameters", style="bold cyan")
        table.add_column("value", style="bold green")
        table.add_column("args", style="bold blue")
        table.add_row("epochs", str(self.config.epochs), "-")
        table.add_row("batch_size", str(self.config.batch_size), "-")
        table.add_row("num_workers", str(self.config.num_workers), "-")
        table.add_row("optimizer", self.config.optimizer, "-")
        table.add_row("scheduler", self.config.scheduler, "-")
        table.add_row("lr0", str(self.config.lr0), "-")
        table.add_row("lr_min", str(self.config.lr_min), "-")
        table.add_row("Backbone", self.config.backbone, dict2str(self.config.backbone_args[self.config.backbone]))
        table.add_row("Neck", self.config.neck, dict2str(self.config.neck_args[self.config.neck]))
        table.add_row("Head", self.config.head, dict2str(self.config.head_args[self.config.head]))
        table.add_row("pos_type", self.config.pos_type, dict2str(self.config.pos_args[self.config.pos_type]))
        table.add_row("pos_loss_type", self.config.pos_loss_type, dict2str(self.config.pos_loss_args[self.config.pos_loss_type]))
        table.add_row("ori_type", self.config.ori_type, dict2str(self.config.ori_args[self.config.ori_type]))
        table.add_row("ori_loss_type", self.config.ori_loss_type, dict2str(self.config.ori_loss_args[self.config.ori_loss_type]))
        table.add_row("ALPHA", str(self.config.ALPHA), "-")
        table.add_row("BETA", str(self.config.BETA), "-")
        table.add_row("CropAndPaste_p", str(self.config.CropAndPaste_p), "-")
        table.add_row("CropAndPadSafe_p", str(self.config.CropAndPadSafe_p), "-")
        table.add_row("DropBlockSafe_p", str(self.config.DropBlockSafe_p), dict2str(self.config.DropBlockSafe_args))
        table.add_row("ZAxisRotation_p", str(self.config.ZAxisRotation_p), dict2str(self.config.ZAxisRotation_args))
        table.add_row("Perspective_p", str(self.config.Perspective_p), dict2str(self.config.Perspective_args))
        table.add_row("AlbumentationAug_p", str(self.config.AlbumentationAug_p), "-")
        rich.print(table)
        # tags
        tags = [self.config.exp_type, self.config.backbone, self.config.neck]
        if self.config.neck == "DensAttFPN":
            tags += ["att_type:" + str(self.config.neck_args["DensAttFPN"]["att_type"])]
        tags += [self.config.head]
        tags += ["pos_type:" + self.config.pos_type, "pos_loss_type:" + self.config.pos_loss_type]
        if self.config.pos_type == "DiscreteSpher":
            tags += [f"angle_stride:{self.config.pos_args['DiscreteSpher']['angle_stride']}",
                     f"r_stride:{self.config.pos_args['DiscreteSpher']['r_stride']}"]
        tags += ["ori_type:" + self.config.ori_type, "ori_loss_type:" + self.config.ori_loss_type]
        if self.config.ori_type == "DiscreteEuler":
            tags += [f"stride:{self.config.ori_args['DiscreteEuler']['stride']}"]
        self.trainer.logger.log_tags(tags)
        # hyperparams
        self.trainer.logger.log_hyperparams(self.config)
        # code
        father_folder = Path("./SPEN")
        for file in father_folder.rglob("*.py"):
            if file.name != "__init__.py":
                self.trainer.logger.log_code(file_path=file)
        # datasetsplit
        father_folder = Path(".").resolve().parent
        dataset_folder = father_folder / "datasets" / "speed"
        self.trainer.logger.log_code(dataset_folder / "train.txt")
        self.trainer.logger.log_code(dataset_folder / "val.txt")
        self.trainer.logger.log_code(dataset_folder / "train_label.json")
        self.trainer.logger.log_code(dataset_folder / "val_label.json")
        # compile
        if self.config.compile:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)


    def forward(self, x):
        return self.model(x)


    def train_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        pos_pre_dict, ori_pre_dict = self.forward(images)
        # loss
        pos_loss_dict = self.pos_loss(pos_pre_dict, labels["pos_encode"])
        ori_loss_dict = self.ori_loss(ori_pre_dict, labels["ori_encode"])
        pos_loss = torch.sum(torch.stack([val for val in pos_loss_dict.values()]))
        ori_loss = torch.sum(torch.stack([val for val in ori_loss_dict.values()]))
        beta_0 = self.beta_0_list[self.trainer.now_epoch-1]
        beta_1 = self.beta_1_list[self.trainer.now_epoch-1]
        train_loss = beta_0 * pos_loss + beta_1 * ori_loss
        if self.config.pos_type == "DiscreteSpher":
            spher_decode = self.discrete_spher2spher(pos_pre_dict)
            spher_loss_dict = self.spher_loss(spher_decode, labels["spher"])
            spher_loss = torch.sum(torch.stack([val for val in spher_loss_dict.values()]))
            train_loss = train_loss + self.BETA[2] * spher_loss
        if self.config.ori_type == "DiscreteEuler":
            euler_decode = self.discrete_euler2euler(ori_pre_dict)
            euler_loss_dict = self.euler_loss(euler_decode, labels["euler"])
            euler_loss = torch.sum(torch.stack([val for val in euler_loss_dict.values()]))
            train_loss = train_loss + self.BETA[3] * euler_loss
        # metrics
        loss_dict = {"pos": pos_loss_dict, "ori": ori_loss_dict}
        if self.config.pos_type == "DiscreteSpher":
            loss_dict["spher"] = spher_loss_dict
        if self.config.ori_type == "DiscreteEuler":
            loss_dict["euler"] = euler_loss_dict
        self._update_train_metrics(num_samples, loss_dict, train_loss, beta_0, beta_1)
        self._train_log(log_online=False)
        return train_loss
    
    
    def on_train_epoch_end(self):
        self._train_log(log_online=True)
        self._train_metrics_reset()
    
    
    def val_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        pos_pre_dict, ori_pre_dict = self.forward(images)
        # metrics
        pos_decode = self.pos_decoder.decode_batch(pos_pre_dict)
        ori_decode = self.ori_decoder.decode_batch(ori_pre_dict)
        self._update_val_metrics(num_samples,
                                 pos_decode, labels["pos"],
                                 ori_decode, labels["ori"])
        self._val_log(log_online=True)
    

    def on_val_epoch_end(self):
        self._val_log(log_online=True)
        self._val_metrics_reset()


    def _loss_init(self, config):
        self.pos_loss = PosLossFunc(config.pos_type, config.pos_loss_type, **config.pos_loss_args[config.pos_loss_type])
        self.ori_loss = OriLossFunc(config.ori_type, config.ori_loss_type, **config.ori_loss_args[config.ori_loss_type])
        self.pos_loss = torch.compile(self.pos_loss)
        self.ori_loss = torch.compile(self.ori_loss)
        if config.pos_type == "DiscreteSpher":
            self.spher_loss = PosLossFunc("Spher", "L1", **config.pos_loss_args["L1"])
            self.spher_loss = torch.compile(self.spher_loss)
        if config.ori_type == "DiscreteEuler":
            self.euler_loss = OriLossFunc("Euler", "L1", **config.ori_loss_args["L1"])
            self.euler_loss = torch.compile(self.euler_loss)

    
    def _metrics_init(self, config):
        self.train_pos_loss = PosLoss(config.pos_type)
        self.train_ori_loss = OriLoss(config.ori_type)
        if config.pos_type == "DiscreteSpher":
            self.train_spher_loss = PosLoss("Spher")
        if config.ori_type == "DiscreteEuler":
            self.train_euler_loss = OriLoss("Euler")
        self.train_loss = Loss()
        self.beta_0 = None
        self.beat_1 = None

        self.pos_error = PosError()
        self.ori_error = OriError()
        self.score = Score(config.ALPHA)


    def _update_train_metrics(self, num_samples: int, loss_dict: Dict[str, Dict[str, Tensor]], loss: Tensor, beta_0: float, beta_1: float):
        self.train_pos_loss.update(loss_dict["pos"], num_samples)
        self.train_ori_loss.update(loss_dict["ori"], num_samples)
        if self.config.pos_type == "DiscreteSpher":
            self.train_spher_loss.update(loss_dict["spher"], num_samples)
        if self.config.ori_type == "DiscreteEuler":
            self.train_euler_loss.update(loss_dict["euler"], num_samples)
        self.train_loss.update(loss, num_samples)
        self.beta_0 = beta_0
        self.beta_1 = beta_1
    

    def _train_log(self, log_online):
        data = {}
        data.update(self.train_pos_loss.compute())
        data.update(self.train_ori_loss.compute())
        if self.config.pos_type == "DiscreteSpher":
            data.update(self.train_spher_loss.compute())
        if self.config.ori_type == "DiscreteEuler":
            data.update(self.train_euler_loss.compute())
        data.update({"loss": self.train_loss.compute()})
        data.update({"beta_0": self.beta_0, "beta_1": self.beta_1})
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=True,
                      prefix="train",
                      log_online=log_online)
    

    def _train_metrics_reset(self):
        self.train_pos_loss.reset()
        self.train_ori_loss.reset()
        if self.config.pos_type == "DiscreteSpher":
            self.train_spher_loss.reset()
        if self.config.ori_type == "DiscreteEuler":
            self.train_euler_loss.reset()
        self.train_loss.reset()
        self.beta_0 = None
        self.beta_1 = None


    def _update_val_metrics(self, num_samples: int,
                                  pos_decode: Tensor, pos_label: Tensor,
                                  ori_decode: Tensor, ori_label: Tensor):
        self.pos_error.update(pos_decode, pos_label, num_samples)
        self.ori_error.update(ori_decode, ori_label, num_samples)
        self.score.update(self.pos_error.compute()[1], self.ori_error.compute())
    

    def _val_log(self, log_online):
        data = {}
        pos_error = self.pos_error.compute()
        data.update({
            "pos_error": pos_error[0],
            "Et": pos_error[1],
            "ori_error": self.ori_error.compute(),
            "score": self.score.compute(),
        })
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=True,
                      prefix="val",
                      log_online=log_online)
    

    def _val_metrics_reset(self):
        self.pos_error.reset()
        self.ori_error.reset()
        self.score.reset()