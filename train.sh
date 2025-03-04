#!/bin/bash

# pos_exp
# python3 train.py --exp_type pos_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --pos_type Cart --pos_loss_type L1 --ALPHA 1 0 --BETA 1 0
# python3 train.py --exp_type pos_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --pos_type Cart --pos_loss_type L2 --ALPHA 1 0 --BETA 1 0
# python3 train.py --exp_type pos_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --pos_type Cart --pos_loss_type SmoothL1 --ALPHA 1 0 --BETA 1 0

# python3 train.py --exp_type pos_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --pos_type Spher --pos_loss_type L1 --ALPHA 1 0 --BETA 1 0
python3 train.py --exp_type pos_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --pos_type Spher --pos_loss_type L2 --ALPHA 1 0 --BETA 1 0
python3 train.py --exp_type pos_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --pos_type Spher --pos_loss_type SmoothL1 --ALPHA 1 0 --BETA 1 0

# ori_exp
# python3 train.py --exp_type ori_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --ori_type Quat --ori_loss_type Cos --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type ori_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --ori_type Quat --ori_loss_type CosDistance --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type ori_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --ori_type Quat --ori_loss_type ExpCos --ALPHA 0 1 --BETA 0 1

# python3 train.py --exp_type ori_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --ori_type Euler --ori_loss_type L1 --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type ori_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --ori_type Euler --ori_loss_type L2 --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type ori_exp --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --ori_type Euler --ori_loss_type SmoothL1 --ALPHA 0 1 --BETA 0 1