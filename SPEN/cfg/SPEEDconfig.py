from pathlib import Path

from ..TorchModel import Config

class SPEEDConfig(Config):
    def __init__(self):
        super().__init__()
        # config
        self.exp_type = "test"
        self.seed = 42
        self.deterministic = False
        self.benchmark = False
        self.debug = False
        self.comet_api = "agcu7oeqU395peWf6NCNqnTa7"
        self.offline = False

        # dataset
        self.dataset_folder = Path("../datasets/speed")
        self.train_ratio = 0.85
        self.val_ratio = 0.15
        self.cache = False
        self.resize_first = False
        self.image_size = (480, 768)

        # train
        self.device = "cuda"
        self.epochs = 300
        self.lr0 = 0.0003
        self.lr_min = 0.00001
        self.warmup_epochs = 3
        self.weight_decay = 0.00001
        self.optimizer = "AdamW"
        self.scheduler = "OnPlateau"              # WarmupCosin, OnPlateau
        self.batch_size = 24
        self.num_workers = 6

        # model
        # backbone
        self.backbone = "mobilenetv4_conv_medium"
        self.backbone_args = {
            "resnet18": {
                "bin_folder" : "resnet18.a1_in1k",
                "backbone_out_channels": [64, 64, 128, 256, 512],
            },
            "resnet34": {
                "bin_folder" : "resnet34.a1_in1k",
                "backbone_out_channels": [64, 64, 128, 256, 512],
            },
            "resnet50": {
                "bin_folder" : "resnet50.a1_in1k",
                "backbone_out_channels": [64, 256, 512, 1024, 2048],
            },
            "mobilenetv4_conv_small": {
                "bin_folder": "mobilenetv4_conv_small.e3600_r256_in1k",
                "backbone_out_channels": [64, 96, 128],
            },
            "mobilenetv4_conv_medium": {
                "bin_folder": "mobilenetv4_conv_medium.e500_r256_in1k",
                "backbone_out_channels": [32, 48, 80, 160, 256],
            },
            "mobilenetv4_conv_large": {
                "bin_folder": "mobilenetv4_conv_large.e600_r384_in1k",
                "backbone_out_channels": [24, 48, 96, 192, 512],
            },
        }
        # neck
        self.neck = "TaileNeck"                  # IdentityNeck, ConvNeck, FPNPAN
        self.neck_args = {
            "IdentityNeck": {},
            "ConvNeck": {"align_channels": 160},
            "TaileNeck": {"align_channels": 960},
            "PAFPN": {"align_channels": 160},
            "BiFPN": {"align_channels": 160},
            "DensAttFPN": {"align_channels": 160, "att_type": None},
        }
        # head
        self.pos_ratio = 0.5
        self.avg_size = (1,) if self.neck == "TaileNeck" else (1, 1, 1)
        
        # pos type
        self.pos_type = "Cart"
        self.pos_args = {
            "Cart": {},
            "Spher": {"r_max": 50},
            "DiscreteSpher": {
                "r_max": 50,
                "r_stride": 1,
                "angle_stride": 5,
                "alpha": 0.0,
                "neighbor": 0,
                "device": "cuda",
            }
        }

        # ori type
        self.ori_type = "Quat"
        self.ori_args = {
            "Quat": {},
            "Euler": {"device": "cuda"},
            "DiscreteEuler": {
                "stride": 5,
                "alpha": 0.0,
                "neighbor": 0,
                "device": "cuda"
            },   
        }
        
        # loss
        ## pos_loss
        self.pos_loss_type = "L1"
        self.pos_loss_args = {
            # cart/spher
            "L1": {"reduction": "mean"},
            "L2": {"reduction": "mean"},
            "SmoothL1": {"reduction": "mean",
                         "beta": 1.0},
            # disceretspher
            "CE": {"reduction": "mean"},
            "KL": {"reduction": "mean"},
            "JS": {},
        }
        
        ## ori_loss
        self.ori_loss_type = "Cos"
        self.ori_loss_args = {
            # quat
            "Cos": {},
            "CosDistance": {},
            "ExpCos": {},
            # Euler
            "L1": {"reduction": "mean"},
            "L2": {"reduction": "mean"},
            "SmoothL1": {"reduction": "mean",
                         "beta": 1.0},
            # discreteeuler
            "CE": {"reduction": "mean"},
            "KL": {"reduction": "mean"},
            "JS": {},
        }

        self.ALPHA = (1, 5)              # score
        self.BETA = (1, 5)               # loss

        # augmentation
        self.ZAxisRotation_p = 0.8
        self.ZAxisRotation_args = {
            "max_angle": 180,
            "max_t": 5,
        }

        self.Perspective_p = 0.0
        self.Perspective_args = {
            "rotation_p": 0.0,
            "max_angle": 90,
            "translation_p": 1.0,
            "max_x": 0.2,
            "max_y": 0.2,
            "scale_p": 1.0,
            "max_scale": 0.2,
            "max_t": 5,
        }

        self.CropAndPaste_p = 0.2

        self.CropAndPadSafe_p = 0.2

        self.DropBlockSafe_p = 0.2
        self.DropBlockSafe_args = {
            "drop_num": 7,
        }

        self.AlbumentationAug_p = 0.01

        self.name = f"{self.exp_type}_{self.backbone}_{self.neck}_{self.pos_type}_{self.pos_loss_type}_{self.ori_type}_{self.ori_loss_type}"