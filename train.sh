#!/usr/bin/env bash

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node 2 \
  --nnodes 1 \
  -m lalaes2.train \
  --conf "wsl.ini" \
  --task "aes2"

wait
