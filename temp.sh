#!/bin/bash
args=()
if [ "$1" = "0" ]
then
  start=11
  end=12
  args+=( '--curriculum' )
  log_name="linear"
elif [ "$1" = "1" ]
then
  start=13
  end=14
  args+=( '--curriculum' )
  log_name="linear"
elif [ "$1" = "2" ]
then
  start=11
  end=12
  log_name="constant"
elif [ "$1" = "3" ]
then
  start=13
  end=14
  log_name="constant"
fi
for ((i=start;i<end;i+=1))
do
 python train_td3.py \
 --max_timesteps 5000000 \
 --batch_size 512 \
 --critic_lr 0.0006 \
 --actor_lr 0.0006 \
 --env robocup_env:robocup-score-v1 \
 --final_scaling 0.4 \
 --log_name "$log_name" \
 --seed $i "${args[@]}"
done
