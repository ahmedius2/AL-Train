#!/bin/bash

CFG_FILE="cfgs/nuscenes_models/mural_pillarnet_01_0128_020.yaml"
CKPT_FILE="../output/nuscenes_models/mural_pillarnet_01_0128_020/default/ckpt/checkpoint_epoch_1.pth"

for RES_IDX in $(seq 0 2)
do
	python test.py --cfg_file $CFG_FILE --ckpt $CKPT_FILE --set MODEL.INF_RES_INDEX $RES_IDX
done
