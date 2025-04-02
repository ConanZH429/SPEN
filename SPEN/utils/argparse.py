import argparse
import time
from ..cfg import SPARKConfig, SPEEDConfig

def parse2config(config):
    parser = argparse.ArgumentParser()
    
    # exp_type
    parser.add_argument("--exp_type", type=str, required=True, help="Experiment type")
    # dataset
    if isinstance(config, SPEEDConfig):
        parser.add_argument("--train_ratio", type=float, default=config.train_ratio, help="Train ratio")
        parser.add_argument("--val_ratio", type=float, default=config.val_ratio, help="Validation ratio")
    parser.add_argument("--cache", action="store_true", help="Cache dataset")
    parser.add_argument("--img_first_size", type=int, nargs="+", default=config.image_first_size, help="Image first size")
    parser.add_argument("--img_size", type=int, nargs="+", default=config.image_size, help="Image size")
    # train
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr0", type=float, default=config.lr0, help="Initial learning rate")
    parser.add_argument("--lr_min", type=float, default=config.lr_min, help="Minimum learning rate")
    parser.add_argument("--beta_cos", action="store_true", help="Beta cosine")
    parser.add_argument("--beta_epochs", type=int, default=config.beta_epochs, help="Beta epochs")
    parser.add_argument("--optimizer", type=str, default=config.optimizer, help="Optimizer")
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Scheduler")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of workers")
    parser.add_argument("--compile", action="store_true", help="Compile")
    parser.add_argument("--gradient_clip_val", type=float, default=config.gradient_clip_val, help="Gradient clip value")
    # backbone
    parser.add_argument("--backbone", "-b", type=str, default=config.backbone, help="Backbone",
                        choices=list(config.backbone_args.keys()))
    # neck
    parser.add_argument("--neck", "-n", type=str, default=config.neck, help="Neck",
                        choices=list(config.neck_args.keys()))
    parser.add_argument("--align_channels", type=int, default=160, help="Align channels")
    parser.add_argument("--att_type", type=str, default=None, help="Attention type",
                        choices=["SSIA", "SE", "SAM", "CBAM"])
    # head
    parser.add_argument("--head", "-h", type=str, default=config.head, help="Head",
                        choices=list(config.head_args.keys()))
    parser.add_argument("--pool_size", type=int, nargs="+", default=(1, ), help="Pool size")
    parser.add_argument("--weighted_learnable", action="store_true", help="Weighted learnable", default=False)
    parser.add_argument("--num_heads", type=int, default=config.head_args["TokenHead"]["num_heads"])
    parser.add_argument("--num_layers", type=int, default=config.head_args["TokenHead"]["num_layers"])
    parser.add_argument("--learnable_token_num", "-ltn", type=int, default=config.head_args["TokenHead"]["learnable_token_num"])
    # pos
    parser.add_argument("--pos_type", "-pt", type=str, default=config.pos_type, help="Position type",
                        choices=list(config.pos_args.keys()))
    parser.add_argument("--r_stride", "-rs", type=float, default=config.pos_args["DiscreteSpher"]["r_stride"], help="r stride")
    parser.add_argument("--angle_stride", "-as", type=float, default=config.pos_args["DiscreteSpher"]["angle_stride"], help="angle stride")
    parser.add_argument("--pos_loss_type", "-plt", type=str, default=config.pos_loss_type, help="Position loss type",
                        choices=list(config.pos_loss_args.keys()))
    # ori
    parser.add_argument("--ori_type", "-ot", type=str, default=config.ori_type, help="Orientation type",
                        choices=list(config.ori_args.keys()))
    parser.add_argument("--euler_stride", "-es", type=int, default=config.ori_args["DiscreteEuler"]["stride"], help="stride")
    parser.add_argument("--ori_loss_type", "-olt", type=str, default=config.ori_loss_type, help="Orientation loss type",
                        choices=list(config.ori_loss_args.keys()))
    # score
    parser.add_argument("--ALPHA", nargs="+", default=config.ALPHA, help="val score alpha")
    # loss beta
    parser.add_argument("--BETA", nargs="+", default=config.BETA, help="loss beta")

    # data augmentation
    parser.add_argument("--zr_p", type=float, default=config.ZAxisRotation_p, help="Z axis rotation probability")
    parser.add_argument("--zr_angle", type=int, default=config.ZAxisRotation_args["max_angle"], help="Z axis rotation angle")
    parser.add_argument("--persp_p", type=float, default=config.Perspective_p)
    parser.add_argument("--persp_angle", type=float, default=config.Perspective_args["max_angle"])
    parser.add_argument("--persp_trans", type=float, default=config.Perspective_args["max_translation"])
    parser.add_argument("--persp_scale", type=float, default=config.Perspective_args["max_scale"])
    parser.add_argument("--crop_paste_p", type=float, default=config.CropAndPaste_p, help="Crop and paste probability")
    parser.add_argument("--crop_pad_p", type=float, default=config.CropAndPadSafe_p, help="Crop and pad safe probability")
    parser.add_argument("--drop_block_p", type=float, default=config.DropBlockSafe_p, help="Drop block safe probability")
    parser.add_argument("--drop_block_num", type=int, default=config.DropBlockSafe_args["drop_num"], help="Drop block safe number of blocks")
    parser.add_argument("--album_p", type=float, default=config.AlbumentationAug_p, help="Albumentations probability")

    args = parser.parse_args()

    # exp_type
    config.exp_type = args.exp_type
    # dataset
    if isinstance(config, SPEEDConfig):
        config.train_ratio = args.train_ratio
        config.val_ratio = args.val_ratio
    config.cache = args.cache
    config.image_first_size = tuple(map(int, args.img_first_size))
    config.image_size = tuple(map(int, args.img_size))
    # train
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr0 = args.lr0
    config.lr_min = args.lr_min
    config.beta_cos = args.beta_cos
    config.beta_epochs = args.beta_epochs
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler
    config.num_workers = args.num_workers
    config.compile = args.compile
    config.gradient_clip_val = args.gradient_clip_val
    # backbone
    config.backbone = args.backbone
    # neck
    config.neck = args.neck
    if config.neck in {"PAFPN", "BiFPN"}:
        config.neck_args[config.neck]["align_channels"] = args.align_channels
    if config.neck in {"DensAttFPN", "TailNeck"}:
        config.neck_args[config.neck]["att_type"] = args.att_type
    # head
    config.head = args.head
    if config.head in {"AvgPoolHead", "MaxPoolHead", "MixPoolHead", "SPPHead", "MHAHead"}:
        config.head_args[config.head]["pool_size"] = tuple(map(int, args.pool_size))
    if config.head in {"MixPoolHead"}:
        config.head_args[config.head]["weighted_learnable"] = args.weighted_learnable
    if config.head in {"TokenHead"}:
        config.head_args[config.head]["num_heads"] = args.num_heads
        config.head_args[config.head]["num_layers"] = args.num_layers
        config.head_args[config.head]["learnable_token_num"] = args.learnable_token_num
    # pos
    config.pos_type = args.pos_type
    if config.pos_type in {"DiscreteSpher"}:
        config.pos_args[config.pos_type]["r_stride"] = args.r_stride
        config.pos_args[config.pos_type]["angle_stride"] = args.angle_stride
    config.pos_loss_type = args.pos_loss_type
    # ori
    config.ori_type = args.ori_type
    if config.ori_type in {"DiscreteEuler"}:
        config.ori_args[config.ori_type]["stride"] = args.euler_stride
    config.ori_loss_type = args.ori_loss_type
    # score
    config.ALPHA = tuple(map(float, args.ALPHA))
    # loss beta
    config.BETA = tuple(map(float, args.BETA))
    # data augmentation
    config.ZAxisRotation_p = args.zr_p
    config.ZAxisRotation_args["max_angle"] = args.zr_angle
    config.Perspective_p = args.persp_p
    config.Perspective_args["max_angle"] = args.persp_angle
    config.Perspective_args["max_translation"] = args.persp_trans
    config.Perspective_args["max_scale"] = args.persp_scale
    config.CropAndPaste_p = args.crop_paste_p
    config.CropAndPadSafe_p = args.crop_pad_p
    config.DropBlockSafe_p = args.drop_block_p
    config.DropBlockSafe_args["drop_num"] = args.drop_block_num
    config.AlbumentationAug_p = args.album_p
    # name
    config.name = f"{config.exp_type}_{config.backbone}_{config.neck}_{config.pos_type}_{config.pos_loss_type}_{config.ori_type}_{config.ori_loss_type}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    return config