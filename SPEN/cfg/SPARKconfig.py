from pathlib import Path

from ..TorchModel import Config

class SPARKConfig(Config):
    def __init__(self):
        super().__init__()
        # config
        self.exp_type = "test"
        self.seed = 9999
        self.deterministic = False
        self.benchmark = True
        self.debug = False
        self.comet_api = "agcu7oeqU395peWf6NCNqnTa7"
        self.offline = False

        # dataset
        self.dataset_folder = Path("../SPARK-Stream-2")
        self.cache = True
        self.resize_first = True
        # 1080 1440
        # self.image_first_size = (1000, 1600)
        self.image_first_size = (720, 960)
        # self.image_size = (432, 576)
        self.image_size = (360, 480)

        # train
        self.device = "cuda"
        self.epochs = 300
        self.lr0 = 0.001
        self.lr_min = 0.000001
        self.warmup_epochs = 5
        self.weight_decay = 0.00001
        self.optimizer = "AdamW"
        self.scheduler = "WarmupCosin"              # WarmupCosin, OnPlateau, ReduceWarmupCosin
        self.batch_size = 40
        self.num_workers = 20
        self.compile = False
        self.gradient_clip_val = None

        # model
        # backbone
        self.backbone = "mobilenetv3_large_100"
        self.backbone_args = {
            "resnet18": {
                "bin_folder" : "resnet18.a1_in1k",
            },
            "mobilenetv3_large_100": {
                "bin_folder" : "mobilenetv3_large_100.miil_in21k",
            },
        }
        # neck
        self.neck = "TaileNeck"                  # IdentityNeck, ConvNeck, FPNPAN
        self.neck_args = {
            "TaileNeck": {},
            "IdentityNeck": {},
            "ConvNeck": {},
            "PAFPN": {"align_channels": 160},
            "BiFPN": {"align_channels": 160},
            "DensAttFPN": {"att_type": None},    # SE, SAM, CBAM, SSIA
        }
        # head
        self.head = "AvgPoolHead"                
        self.head_args = {
            "AvgPoolHead": {"pool_size": (1, )},
            "MaxPoolHead": {"pool_size": (1, )},
            "MixPoolHead": {"pool_size": (1, ),
                            "weighted_learnable": False},
            "SPPHead": {"pool_size": ((1, 2), ),
                        "mode": "max"},
            "MHAHead": {
                "patch_size": (None, ),
                "embedding_mode": "max",
                "pool_size": (1, ),
                "pool_mode": "max",
                "num_heads": 8,
            },
            "TokenHead": {
                "patch_size": (None, ),
                "embedding_mode": "max",
                "num_heads": 8,
                "num_layers": 5,
            }
        }
        
        # pos type
        self.pos_type = "Cart"
        self.pos_args = {
            "Cart": {},
            "Spher": {},
            "DiscreteSpher": {
                "r_max": 50,
                "r_stride": 1,
                "angle_stride": 1,
                "alpha": 0.0,
                "neighbor": 0,
                "device": "cuda",
            }
        }

        # ori type
        self.ori_type = "Quat"
        self.ori_args = {
            "Quat": {},
            "Euler": {},
            "DiscreteEuler": {
                "stride": 1,
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
            "L1": {"reduction": "mean"},
            "CE": {},
            "KL": {},
            "JS": {},
            "WassersteinLoss": {}
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
            "L1": {"reduction": "mean"},
            "CE": {},
            "KL": {},
            "JS": {},
            "WassersteinLoss": {}
        }

        self.ALPHA = (1, 1)              # score
        self.BETA = (1, 1)               # loss

        # augmentation
        self.ZAxisRotation_p = 0.8
        self.ZAxisRotation_args = {
            "max_angle": 180,
            "max_t": 7,
        }

        self.Perspective_p = 0.0
        self.Perspective_args = {
            "rotation_p": 0.1,
            "max_angle": 10,
            "translation_p": 0.1,
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

        self.name = ""