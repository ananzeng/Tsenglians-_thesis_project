#!/bin/sh
EXP=Official_exp_voc_test_ori_DRSlayer_67_learnable_seblock_CDA+_2
TYPE=ms

CUDA_VISIBLE_DEVICES=0 python3 gen_gt.py \
   --dataset=mscoco \
   --datalist=data/voc12/train_aug.txt \
   --gt_dir=./data/voc12/JPEGImages/ \
   --save_path=./data/voc12/pseudo_seg_labels_${EXP}/ \
   --pred_dir=./runs/${EXP}/${TYPE}/attention/ \
   --num_workers=16