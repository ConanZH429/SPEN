#!/bin/bash

# test
# python3 train.py --exp_type test --epochs 100 --batch_size 50 --num_workers 15 --backbone resnet18 --neck TaileNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type Cos --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type best_test --epochs 300 --batch_size 50 --num_workers 15 --backbone mobilenetv4_conv_medium --neck DensAttFPN --att_type SSIA --pos_type DiscreteSpher --pos_loss_type CE --ori_type DiscreteEuler --ori_loss_type CE --avg_size 4 2 1 --ALPHA 1 5 --BETA 1 5
# python3 train.py --cache --exp_type test --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --r_stride 1 --angle_stride 2 --ALPHA 1 0 --BETA 1 0

# pos_exp
# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type L1 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type L2 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --pos_type Cart --pos_loss_type SmoothL1 --ALPHA 1 0 --BETA 1 0

# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --pos_type Spher --pos_loss_type L1 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --pos_type Spher --pos_loss_type L2 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --pos_type Spher --pos_loss_type SmoothL1 --ALPHA 1 0 --BETA 1 0

# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --ALPHA 1 0 --BETA 1 0

# ori_exp
# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --ori_type Quat --ori_loss_type Cos --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --ori_type Quat --ori_loss_type CosDistance --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --ori_type Quat --ori_loss_type ExpCos --ALPHA 0 1 --BETA 0 1

# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --ori_type Euler --ori_loss_type L1 --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --ori_type Euler --ori_loss_type L2 --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 1 --ori_type Euler --ori_loss_type SmoothL1 --ALPHA 0 1 --BETA 0 1

# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type KL --ALPHA 0 1 --BETA 0 1

# discrete_pos_exp
## angle_stride 2, 5, 10
### loss_type CE
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 2 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 5 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --ALPHA 1 0 --BETA 1 0
### loss_type KL
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --r_stride 1 --angle_stride 2 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --r_stride 1 --angle_stride 5 --ALPHA 1 0 --BETA 1 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --r_stride 1 --angle_stride 10 --ALPHA 1 0 --BETA 1 0
## r_stride 2 5 angle_stride=10
### loss_type CE
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 2 --angle_stride 10 --ALPHA 1 0 --BETA 1.5 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 5 --angle_stride 10 --ALPHA 1 0 --BETA 1.5 0
## r_stride=1 angle_stride=10, neighbor=1, alpha=0.1 0.2 0.3
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --discrete_spher_neighbor 1 --discrete_spher_alpha 0.1 --ALPHA 1 0 --BETA 1.5 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --discrete_spher_neighbor 1 --discrete_spher_alpha 0.2 --ALPHA 1 0 --BETA 1.5 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --discrete_spher_neighbor 1 --discrete_spher_alpha 0.3 --ALPHA 1 0 --BETA 1.5 0
## r_stride=1 angle_stride=10, alpha=0.1, neighbor=2 3 4
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --discrete_spher_neighbor 2 --discrete_spher_alpha 0.1 --ALPHA 1 0 --BETA 1.5 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --discrete_spher_neighbor 3 --discrete_spher_alpha 0.1 --ALPHA 1 0 --BETA 1.5 0
# python3 train.py --cache --exp_type discrete_pos_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --discrete_spher_neighbor 4 --discrete_spher_alpha 0.1 --ALPHA 1 0 --BETA 1.5 0



# discrete_ori_exp
## stride 2, 5, 10
### loss_type CE
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 2 --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 10 --ALPHA 0 1 --BETA 0 1
### loss_type KL
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type KL --stride 2 --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type KL --stride 5 --ALPHA 0 1 --BETA 0 1
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type KL --stride 10 --ALPHA 0 1 --BETA 0 1
## stride=5, neighbor=1, alpha=0.1 0.2 0.3
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.1 --ALPHA 0 1 --BETA 0 1.2
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.2 --ALPHA 0 1 --BETA 0 1.2
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.3 --ALPHA 0 1 --BETA 0 1.2
## stride=5, alpha=0.3, neighbor=2 3 4
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 2 --discrete_euler_alpha 0.3 --ALPHA 0 1 --BETA 0 1.2
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 3 --discrete_euler_alpha 0.3 --ALPHA 0 1 --BETA 0 1.2
# python3 train.py --cache --exp_type discrete_ori_exp --epochs 100 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 4 --discrete_euler_alpha 0.3 --ALPHA 0 1 --BETA 0 1.2


python3 train.py --cache --exp_type combine --epochs 300 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.3 --ALPHA 1 1 --BETA 0.1 0.9
python3 train.py --cache --exp_type combine --epochs 300 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.3 --ALPHA 1 1 --BETA 0.2 0.8
python3 train.py --cache --exp_type combine --epochs 300 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.3 --ALPHA 1 1 --BETA 0.3 0.7
python3 train.py --cache --exp_type combine --epochs 300 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.3 --ALPHA 1 1 --BETA 0.4 0.6
python3 train.py --cache --exp_type combine --epochs 300 --batch_size 60 --num_workers 15 --backbone resnet18 --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type CE --r_stride 1 --angle_stride 10 --ori_type DiscreteEuler --ori_loss_type CE --stride 5 --discrete_euler_neighbor 1 --discrete_euler_alpha 0.3 --ALPHA 1 1 --BETA 0.5 0.5
