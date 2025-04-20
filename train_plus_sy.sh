#!/bin/bash

# backbone
python3 train_plus_sy.py --exp_type test_plus --cache --compile --lr0 0.001 --scheduler WarmupCosin --img_size 480 768 --beta_cos --beta_epochs 300 --epochs 300 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train_plus_sy.py --exp_type test_plus --cache --compile --lr0 0.001 --scheduler WarmupCosin --img_size 480 768  --beta_cos --beta_epochs 50 --epochs 50 --batch_size 40 -b efficientnet_b3 -n TailNeck --head TokenHead --num_heads 8 --num_layers 12 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
/usr/bin/shutdown
