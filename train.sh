#!/bin/bash

# test
# python3 train.py --exp_type test --epochs 300 --batch_size 50 --num_workers 15 --backbone mobilenetv4_conv_small --neck TaileNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type Cos --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type best_test --epochs 300 --batch_size 50 --num_workers 15 --backbone mobilenetv4_conv_medium --neck DensAttFPN --att_type SSIA --pos_type DiscreteSpher --pos_loss_type CE --ori_type DiscreteEuler --ori_loss_type CE --avg_size 4 2 1 --ALPHA 1 5 --BETA 1 5
# python3 train.py --cache --exp_type test --epochs 300 --batch_size 60 --num_workers 15 --backbone mobilenetv4_conv_small --neck TaileNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --r_stride 1 --angle_stride 2 --ALPHA 1 0 --BETA 1 0

# cart+quat
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type CosDistance --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type ExpCos --ALPHA 1 1 --BETA 1 1

# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Cart --pos_loss_type L2 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Cart --pos_loss_type SmoothL1 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1

# spher+Euler
# python3 train.py --cache --exp_type SpherAndEuler --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Spher --pos_loss_type L1 --ori_type Euler --ori_loss_type L1 --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type SpherAndEuler --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Spher --pos_loss_type L2 --ori_type Euler --ori_loss_type L2 --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type SpherAndEuler --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type Spher --pos_loss_type SmoothL1 --ori_type Euler --ori_loss_type SmoothL1 --ALPHA 1 1 --BETA 1 1

# Discrete
# python3 train.py --cache --exp_type Discrete --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type DiscreteSpher --pos_loss_type CE --ori_type DiscreteEuler --ori_loss_type CE --ALPHA 1 1 --BETA 1 1 1 1
# python3 train.py --cache --exp_type Discrete --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --pos_type DiscreteSpher --pos_loss_type KL --ori_type DiscreteEuler --ori_loss_type KL --ALPHA 1 1 --BETA 1 1 1 1

# BETA
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 1 1 1
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 3 1 3
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 5 1 5
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 7 1 7

# BETA_COS
# python3 train.py --exp_type BETA --cache --compile --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 5 1 5


# DiscreteHyper
## euler stride
# python3 train.py --exp_type EulerStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 2 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type EulerStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type EulerStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 10 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5

## spher angle stride
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 0.5 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 2 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 5 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 10 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5

## r stride
# python3 train.py --exp_type RStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 0.5 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type RStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 2 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5

# Perspective Augmentation
# python3 train.py --exp_type Perspective --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5


# neck
# python3 train.py --exp_type Neck --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck PAFPN --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck BiFPN --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.0005 --lr_min 0.000005 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type SE --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.0005 --lr_min 0.000005 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type CBAM --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.0005 --lr_min 0.000005 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type SAM --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type SSIA --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5

# head
# python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head MixPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TaileNeck --head TokenHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
