#!/bin/bash
args=()
if [ "$1" = "0" ]
then
  start=0
  end=5
  args+=( '--curriculum' )
  log_name="curriculum"
elif [ "$1" = "1" ]
then
  start=5
  end=10
  args+=( '--curriculum' )
  log_name="curriculum"
elif [ "$1" = "2" ]
then
  start=0
  end=5
  log_name="no_curriculum"
elif [ "$1" = "3" ]
then
  start=5
  end=10
  log_name="no_curriculum"
fi
echo "$log_name"
for ((i=start;i<end;i+=1))
do
 python train_td3.py \
 --max_timesteps 10000000 \
 --batch_size 512 \
 --critic_lr 0.0006 \
 --actor_lr 0.0006 \
 --env robocup_env:robocup-score-v0 \
 --log_name "$log_name" \
 --seed $i "${args[@]}"
done
