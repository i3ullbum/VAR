#!/bin/bash

torchrun --nproc_per_node=2 --nnodes=1 train.py \
  --depth=16 \
  --bs=192 \
  --ep=200 \
  --fp16=1 \
  --alng=1e-3 \
  --wpe=0.1 \
  --data_path=/data/rlwrld-common/beom/imagenet1k \