#!/bin/bash
python mc.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=/home/eli/code/FrozenBiLM/zssiq2 --suffix="." --batch_size_val=1 --max_tokens=128 --load=/home/eli/code/FrozenBiLM/model/ckpt/frozenbilm_tvqa.pth