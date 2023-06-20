#!/usr/bin/bash

layer=18
model=resnet

python moco/main_moco.py \
  -a "${model}${layer}" \
  --lr 0.015 \
  --batch-size 64 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /path/to/dataset/

# model=vit
# size=base

# python moco_v3/main_moco.py \
#   -a "${model}_${size}" -b 32 \
#   --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
#   --epochs=300 --warmup-epochs=40 \
#   --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
#   --dist-url 'tcp://localhost:10001' \
#   --multiprocessing-distributed --world-size 1 --rank 0 \
#   /path/to/dataset/
