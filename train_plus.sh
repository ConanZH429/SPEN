#!/bin/bash

# backbone
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
python3 train_plus.py --exp_type test_plus --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 50 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9