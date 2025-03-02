#!/bin/bash

mkdir -p tmp_results

CFG_FILE="cfgs/nuscenes_models/mural_pillarnet_016_020_032.yaml"
CKPT_DIR="../output/nuscenes_models/mural_pillarnet_016_020_032/default/ckpt/" #checkpoint_epoch_1.pth"

NUM_CKPT=$(ls -al $CKPT_DIR/checkpoint_epoch_* | wc | cut -d ' ' -f 7)
LAST_RES_IDX=2

for EPOCH in $(seq 1 $NUM_CKPT)
do
	CKPT_FILE=${CKPT_DIR}"checkpoint_epoch_${EPOCH}.pth"
	for RES_IDX in $(seq 0 $LAST_RES_IDX)
	do
		OUT_FILE="./tmp_results/epoch"${EPOCH}"_res"${RES_IDX}".log"
		if [ -f $OUT_FILE ]; then
			printf "Skipping $OUT_FILE\n"
		else
			printf "Running test, output file is: ${OUT_FILE}\n"
			python test.py --cfg_file $CFG_FILE --ckpt $CKPT_FILE \
				--set MODEL.INF_RES_INDEX $RES_IDX > $OUT_FILE
			rm -rf $CKPT_DIR"../eval" # save space ! 
		fi
	done
done
