import argparse
import time

def parse2config(config):
    parser = argparse.ArgumentParser()
    
    # exp_type
    parser.add_argument("--exp_type", type=str, required=True, help="Experiment type")
    # train
    parser.add_argument("--train_ratio", type=float, default=config.train_ratio, help="Train ratio")
    parser.add_argument("--val_ratio", type=float, default=config.val_ratio, help="Validation ratio")
    parser.add_argument("--img_first_size", type=int, nargs="+", default=config.image_first_size, help="Image first size")
    parser.add_argument("--img_size", type=int, nargs="+", default=config.image_size, help="Image size")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs")
    parser.add_argument("--beta_cos", action="store_true", help="Beta cosine")
    parser.add_argument("--beta_epochs", type=int, default=config.beta_epochs, help="Beta epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of workers")
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Scheduler")
    parser.add_argument("--optimizer", type=str, default=config.optimizer, help="Optimizer")
    parser.add_argument("--cache", action="store_true", help="Cache dataset")
    parser.add_argument("--lr0", type=float, default=config.lr0, help="Initial learning rate")
    parser.add_argument("--lr_min", type=float, default=config.lr_min, help="Minimum learning rate")
    parser.add_argument("--compile", action="store_true", help="Compile")
    parser.add_argument("--gradient_clip_val", type=float, default=config.gradient_clip_val, help="Gradient clip value")
    # backbone
    parser.add_argument("--backbone", type=str, default=config.backbone, help="Backbone",
                        choices=list(config.backbone_args.keys()))
    # neck
    parser.add_argument("--neck", type=str, default=config.neck, help="Neck",
                        choices=list(config.neck_args.keys()))
    parser.add_argument("--align_channels", type=int, default=160, help="Align channels")
    parser.add_argument("--att_type", type=str, default=config.neck_args["DensAttFPN"]["att_type"], help="Attention type",
                        choices=["SSIA", "SE", "SAM", "CBAM"])
    # head
    parser.add_argument("--head", type=str, default=config.head, help="Head",
                        choices=list(config.head_args.keys()))
    parser.add_argument("--pool_size", type=int, nargs="+", default=(1, ), help="Pool size")
    parser.add_argument("--weighted_learnable", action="store_true", help="Weighted learnable", default=False)
    # pos
    parser.add_argument("--pos_type", type=str, default=config.pos_type, help="Position type",
                        choices=list(config.pos_args.keys()))
    parser.add_argument("--r_stride", type=float, default=config.pos_args["DiscreteSpher"]["r_stride"], help="r stride")
    parser.add_argument("--angle_stride", type=float, default=config.pos_args["DiscreteSpher"]["angle_stride"], help="angle stride")
    parser.add_argument("--pos_loss_type", type=str, default=config.pos_loss_type, help="Position loss type",
                        choices=list(config.pos_loss_args.keys()))
    # ori
    parser.add_argument("--ori_type", type=str, default=config.ori_type, help="Orientation type",
                        choices=list(config.ori_args.keys()))
    parser.add_argument("--stride", type=int, default=config.ori_args["DiscreteEuler"]["stride"], help="stride")
    parser.add_argument("--ori_loss_type", type=str, default=config.ori_loss_type, help="Orientation loss type",
                        choices=list(config.ori_loss_args.keys()))
    # score
    parser.add_argument("--ALPHA", nargs="+", default=config.ALPHA, help="val score alpha")
    # loss beta
    parser.add_argument("--BETA", nargs="+", default=config.BETA, help="loss beta")

    # data augmentation
    parser.add_argument("--Zr_angle", type=int, default=config.ZAxisRotation_args["max_angle"], help="Z axis rotation angle")

    args = parser.parse_args()

    config.exp_type = args.exp_type
    config.train_ratio = args.train_ratio
    config.val_ratio = args.val_ratio
    config.image_first_size = tuple(map(int, args.img_first_size))
    config.image_size = tuple(map(int, args.img_size))
    config.epochs = args.epochs
    config.beta_cos = args.beta_cos
    config.beta_epochs = args.beta_epochs
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.scheduler = args.scheduler
    config.optimizer = args.optimizer
    config.cache = args.cache
    config.lr0 = args.lr0
    config.lr_min = args.lr_min
    config.compile = args.compile
    config.gradient_clip_val = args.gradient_clip_val
    config.backbone = args.backbone
    config.neck = args.neck
    config.neck_args["PAFPN"]["align_channels"] = args.align_channels
    config.neck_args["BiFPN"]["align_channels"] = args.align_channels
    config.neck_args["DensAttFPN"]["att_type"] = args.att_type
    config.head = args.head
    config.head_args[config.head]["pool_size"] = tuple(map(int, args.pool_size))
    if config.head == "SPPHead":
        config.head_args["SPPHead"]["pool_size"] = ((1, 2), )
    config.head_args["MixPoolHead"]["weighted_learnable"] = args.weighted_learnable
    config.pos_type = args.pos_type
    config.pos_args["DiscreteSpher"]["r_stride"] = args.r_stride
    config.pos_args["DiscreteSpher"]["angle_stride"] = args.angle_stride
    config.pos_loss_type = args.pos_loss_type
    config.ori_type = args.ori_type
    config.ori_args["DiscreteEuler"]["stride"] = args.stride
    config.ori_loss_type = args.ori_loss_type
    config.ALPHA = tuple(map(float, args.ALPHA))
    config.BETA = tuple(map(float, args.BETA))
    config.ZAxisRotation_args["max_angle"] = args.Zr_angle
    config.name = f"{config.exp_type}_{config.backbone}_{config.neck}_{config.pos_type}_{config.pos_loss_type}_{config.ori_type}_{config.ori_loss_type}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    return config