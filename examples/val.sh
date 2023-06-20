#!/usr/bin/bash

layer=18
model=resnet

python moco/main_lincls.py \
  -a "${model}${layer}" \
  --lr 30.0 \
  --batch-size 512 \
  --pretrained "ckpt/${model}${layer}/checkpoint_0180.pth.tar" \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/dataset/

# model=vit
# size=base
# python moco_v3/main_lincls.py \
#   -a "${model}_${size}" -b 2048 --lr 30.0 \
#   --dist-url 'tcp://localhost:10001' \
#   --multiprocessing-distributed --world-size 1 --rank 0 \
#   --pretrained "ckpt/${model}_${size}/checkpoint_0292.pth.tar" \
#   /path/to/dataset/
