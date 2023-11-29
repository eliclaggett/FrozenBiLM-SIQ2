#!/bin/bash
#python mc.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=/home/eli/code/FrozenBiLM/zssiq2 --suffix="." --batch_size_val=1 --max_tokens=128 --load=/home/eli/code/FrozenBiLM/model/ckpt/frozenbilm_tvqa.pth

python mc.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=/home/eli/code/FrozenBiLM/zssiq2-d3 --suffix="." --batch_size_val=1 --max_tokens=128 --load=/home/eli/code/FrozenBiLM/zssiq2/best_model.pth
#python -m torch.distributed.launch --nproc_per_node 1 --use_env mc_clip.py --test --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=zsclip --batch_size_val=8 --max_feats=1clear
