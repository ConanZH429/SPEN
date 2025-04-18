from pathlib import Path

from ..TorchModel import Config

class SPEEDplusConfig(Config):
    def __init__(self):
        super().__init__()
        # config
        self.exp_type = "test"
        self.seed = 9999
        self.benchmark = True
        self.debug = False
        self.comet_api = "agcu7oeqU395peWf6NCNqnTa7"
        self.offline = False

        # dataset
        self.dataset_folder = Path("../speedplusv2")
        self.cache = True
        self.resize_first = True
        self.image_first_size = (1000, 1600)
        self.image_first_size = (900, 1440)
        # self.image_first_size = (800, 1280)
        self.image_size = (480, 768)
        # self.image_size = (400, 640)

        # train
        self.device = "cuda"
        self.epochs = 400
        self.batch_size = 50
        self.lr0 = 0.001
        self.lr_min = 0.000001
        self.warmup_epochs = 5
        self.beta_cos = False
        self.beta_epochs = 400
        self.weight_decay = 0.00001
        self.optimizer = "AdamW"
        self.scheduler = "WarmupCosin"              # WarmupCosin, OnPlateau, ReduceWarmupCosin, MultiStepLR
        self.num_workers = 20
        self.compile = False
        self.gradient_clip_val = None

        # model
        # backbone
        self.backbone = "mobilenetv3_large_100"
        self.pretrained = True
        self.backbone_args = {
            "mobilenetv3_small_075": {
                "bin_folder": "mobilenetv3_small_075.lamb_in1k",
                "out_channels": [16, 24, 32, 40, 72],
            },
            "mobilenetv3_small_100": {
                "bin_folder": "mobilenetv3_small_100.lamb_in1k",
                "out_channels": [16, 24, 40, 48, 96]
            },
            "mobilenetv3_large_075": {
                "bin_folder": "tf_mobilenetv3_large_075.in1k",
                "out_channels": [16, 24, 32, 64, 88, 120]
            },
            "mobilenetv3_large_100": {
                "bin_folder": "mobilenetv3_large_100.miil_in21k",
                "out_channels": [16, 24, 40, 80, 112, 160]
            },
            "mobilenetv3_large_150d": {
                "bin_folder": "mobilenetv3_large_150d.ra4_e3600_r256_in1k",
                "out_channels": [24, 40, 64, 120, 168, 240]
            },
            "resnet34d": {
                "bin_folder": "resnet34d.ra2_in1k",
                "out_channels": [64, 64, 128, 256, 512]
            }
        }
        # neck
        self.neck = "TailNeck"                  # IdentityNeck, ConvNeck, FPNPAN
        self.neck_args = {
            "TailNeck": {
                "att_type": None
            },
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
                        "mode": "mean"},
            "MHAHead": {
                "patch_size": (None, ),
                "embedding_mode": "mean",
                "pool_size": (1, ),
                "pool_mode": "mean",
                "num_heads": 8,
            },
            "TokenHead": {
                "patch_size": (None, ),
                "embedding_mode": "mean",
                "num_heads": 8,
                "num_layers": 8,
                "learnable_token_num": 6
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
        }

        self.ALPHA = (5, 1)              # score
        self.BETA = (1, 5)               # loss

        # augmentation
        self.ZAxisRotation_p = 0.0
        self.ZAxisRotation_args = {
            "max_angle": 180,
            "max_t": 7,
        }

        self.Perspective_p = 0.0
        self.Perspective_args = {
            "rotation_p": 1.0,
            "max_angle": 10,
            "translation_p": 1.0,
            "max_translation": 0.1,
            "scale_p": 1.0,
            "max_scale": 0.1,
            "max_t": 5,
        }

        self.CropAndPaste_p = 0.0

        self.CropAndPadSafe_p = 0.0

        self.DropBlockSafe_p = 0.0
        self.DropBlockSafe_args = {
            "drop_num": 5,
        }

        self.AlbumentationAug_p = 0.5

        self.name = ""