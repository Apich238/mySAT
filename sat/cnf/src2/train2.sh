#!/bin/bash
cdate=$(date +%Y-%m-%d_%H-%M-%S)
echo $cdate
docker run --rm -it -v /mnt:/mnt -w /mnt/nvme208/alex/sat/src --gpus 'device=8' --user 1000:1000 \
  --name sat_training_${cdate} speaker_id_image python3 experiments2.py