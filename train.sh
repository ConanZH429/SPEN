#!/bin/bash

# test
# python3 train.py --exp_type test --epochs 300 --batch_size 50 --num_workers 15 --backbone mobilenetv4_conv_small --neck TaileNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type Cos --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type best_test --epochs 300 --batch_size 50 --num_workers 15 --backbone mobilenetv4_conv_medium --neck DensAttFPN --att_type SSIA --pos_type DiscreteSpher --pos_loss_type CE --ori_type DiscreteEuler --ori_loss_type CE --avg_size 4 2 1 --ALPHA 1 5 --BETA 1 5
# python3 train.py --cache --exp_type test --epochs 300 --batch_size 60 --num_workers 15 --backbone mobilenetv4_conv_small --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --r_stride 1 --angle_stride 2 --ALPHA 1 0 --BETA 1 0

# cart+quat
python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1
python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type CosDistance --ALPHA 1 1 --BETA 1 1
python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type ExpCos --ALPHA 1 1 --BETA 1 1

python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type L2 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1
python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type SmoothL1 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1

# Discrete
python3 train.py --cache --exp_type Discrete --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --avg_size 1 --pos_type DiscreteSpher --pos_loss_type CE --ori_type DiscreteEuler --ori_loss_type CE --ALPHA 1 1 --BETA 1 1 1 1
python3 train.py --cache --exp_type Discrete --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --avg_size 1 --pos_type DiscreteSpher --pos_loss_type KL --ori_type DiscreteEuler --ori_loss_type KL --ALPHA 1 1 --BETA 1 1 1 1