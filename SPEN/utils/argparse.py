import argparse
import time

def parse2config(config):
    parser = argparse.ArgumentParser()
    
    # exp_type
    parser.add_argument("--exp_type", type=str, required=True, help="Experiment type")
    # train
    parser.add_argument("--train_ratio", type=float, default=config.train_ratio, help="Train ratio")
    parser.add_argument("--val_ratio", type=float, default=config.val_ratio, help="Validation ratio")
    parser.add_argument("--img_size", type=int, nargs="+", default=config.image_size, help="Image size")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of workers")
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Scheduler")
    parser.add_argument("--optimizer", type=str, default=config.optimizer, help="Optimizer")
    parser.add_argument("--cache", action="store_true", help="Cache dataset")
    parser.add_argument("--lr0", type=float, default=config.lr0, help="Initial learning rate")
    parser.add_argument("--compile", action="store_true", help="Compile")
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
    parser.add_argument("--weighted", action="store_true", help="Weighted")
    parser.add_argument("--avg_size", type=int, nargs="+", default=config.avg_size, help="global average pool size")
    # pos
    parser.add_argument("--pos_type", type=str, default=config.pos_type, help="Position type",
                        choices=list(config.pos_args.keys()))
    parser.add_argument("--r_stride", type=int, default=config.pos_args["DiscreteSpher"]["r_stride"], help="r stride")
    parser.add_argument("--angle_stride", type=int, default=config.pos_args["DiscreteSpher"]["angle_stride"], help="angle stride")
    parser.add_argument("--discrete_spher_alpha", type=float, default=config.pos_args["DiscreteSpher"]["alpha"], help="discrete spher alpha")
    parser.add_argument("--discrete_spher_neighbor", type=int, default=config.pos_args["DiscreteSpher"]["neighbor"], help="discrete spher neighbor")
    parser.add_argument("--pos_loss_type", type=str, default=config.pos_loss_type, help="Position loss type",
                        choices=list(config.pos_loss_args.keys()))
    # ori
    parser.add_argument("--ori_type", type=str, default=config.ori_type, help="Orientation type",
                        choices=list(config.ori_args.keys()))
    parser.add_argument("--stride", type=int, default=config.ori_args["DiscreteEuler"]["stride"], help="stride")
    parser.add_argument("--discrete_euler_alpha", type=float, default=config.ori_args["DiscreteEuler"]["alpha"], help="discrete euler alpha")
    parser.add_argument("--discrete_euler_neighbor", type=int, default=config.ori_args["DiscreteEuler"]["neighbor"], help="discrete euler neighbor")
    parser.add_argument("--ori_loss_type", type=str, default=config.ori_loss_type, help="Orientation loss type",
                        choices=list(config.ori_loss_args.keys()))
    # score
    parser.add_argument("--ALPHA", nargs="+", default=config.ALPHA, help="val score alpha")
    # loss beta
    parser.add_argument("--BETA", nargs="+", default=config.BETA, help="loss beta")

    args = parser.parse_args()

    config.exp_type = args.exp_type
    config.train_ratio = args.train_ratio
    config.val_ratio = args.val_ratio
    config.image_size = tuple(map(int, args.img_size))
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.scheduler = args.scheduler
    config.optimizer = args.optimizer
    config.cache = args.cache
    config.lr0 = args.lr0
    config.compile = args.compile
    config.backbone = args.backbone
    config.neck = args.neck
    config.neck_args["PAFPN"]["align_channels"] = args.align_channels
    config.neck_args["BiFPN"]["align_channels"] = args.align_channels
    config.neck_args["DensAttFPN"]["att_type"] = args.att_type
    config.weighted = args.weighted
    config.avg_size = tuple(map(int, args.avg_size))
    config.pos_type = args.pos_type
    config.pos_args["DiscreteSpher"]["r_stride"] = args.r_stride
    config.pos_args["DiscreteSpher"]["angle_stride"] = args.angle_stride
    config.pos_args["DiscreteSpher"]["alpha"] = args.discrete_spher_alpha
    config.pos_args["DiscreteSpher"]["neighbor"] = args.discrete_spher_neighbor
    config.pos_loss_type = args.pos_loss_type
    config.ori_type = args.ori_type
    config.ori_args["DiscreteEuler"]["stride"] = args.stride
    config.ori_args["DiscreteEuler"]["alpha"] = args.discrete_euler_alpha
    config.ori_args["DiscreteEuler"]["neighbor"] = args.discrete_euler_neighbor
    config.ori_loss_type = args.ori_loss_type
    config.ALPHA = tuple(map(float, args.ALPHA))
    config.BETA = tuple(map(float, args.BETA))
    config.name = f"{config.exp_type}_{config.backbone}_{config.neck}_{config.pos_type}_{config.pos_loss_type}_{config.ori_type}_{config.ori_loss_type}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    return config