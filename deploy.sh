#!/bin/bash

SRC=/home/eli/code/FrozenBiLM
DST=/mnt/d/deploy.tar.gz

# Make tar.gz archive
echo "Compressing files..."
tar -zcf $DST -C $SRC requirements.txt run_eval.sh run_tune.sh mc.py args.py main.py model datasets util deberta-v2-xlarge