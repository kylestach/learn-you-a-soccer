#!/bin/bash

# Script to reproduce results

for ((i=100;i<110;i+=1))
do
	python train_td3.py \
	--policy "TD3" \
	--seed $i \
	--save_model \
  --max_timesteps 10000000
done
