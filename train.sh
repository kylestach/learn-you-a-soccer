#!/bin/bash

for ((i=0;i<10;i+=1))
do
 python train_td3.py \
 --max_timesteps 10000000 \
 --batch_size 512 \
 --critic_lr 0.0006 \
 --actor_lr 0.0006 \
 --env robocup_env:robocup-score-v0 \
 --log_name "curriculum" \
 --seed $i \
 --curriculum
done

for ((i=0;i<10;i+=1))
do
 python train_td3.py \
 --max_timesteps 10000000 \
 --batch_size 512 \
 --critic_lr 0.0006 \
 --actor_lr 0.0006 \
 --env robocup_env:robocup-score-v0 \
 --log_name "no_curriculum" \
 --seed $i
done
