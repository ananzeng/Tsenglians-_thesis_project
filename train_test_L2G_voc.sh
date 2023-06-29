#!/bin/sh
#完整版 還未測試!
EXP=Official_exp_voc_test_ori_DRSlayer_67_learnable_seblock_CDA+_lr0.0005_tr0.2
RUN_FILE=train_l2g_sal_voc.py
BASH_FILE=train_test_L2G_voc.sh
GPU_ID=0
CROP_SIZE=320
PATCH_NUM=6

mkdir -p runs/${EXP}/model/
cp ${BASH_FILE} runs/${EXP}/model/${BASH_FILE}
cp scripts/${RUN_FILE} runs/${EXP}/model/${RUN_FILE} 
cp models/resnet38_base.py runs/${EXP}/model/
cp models/resnet38.py runs/${EXP}/model/
cp utils/LoadData.py runs/${EXP}/model/
chmod 444 runs/${EXP}/model/
#lr 0.001
#--epoch=10 
#--epoch=13
CUDA_VISIBLE_DEVICES=2,3 python3 ./scripts/${RUN_FILE} \
    --img_dir=./data/voc12/ \
    --train_list=./data/voc12/train_cls.txt \
    --test_list=./data/voc12/val_cls.txt \
    --epoch=13 \
    --lr=0.0005 \
    --batch_size=3 \
    --iter_size=1 \
    --dataset=pascal_voc \
    --input_size=448 \
    --crop_size=${CROP_SIZE} \
    --disp_interval=100 \
    --num_classes=20 \
    --num_workers=8 \
    --patch_size=${PATCH_NUM} \
    --snapshot_dir=./runs/${EXP}/model/  \
    --att_dir=./runs/${EXP}/  \
    --decay_points='5' \
    --kd_weights=10 \
    --bg_thr=0.001 \
    --cda \
    --cda_tr=0.2 \
    --load_checkpoint="./runs/Official_exp_voc_test_ori_DRSlayer_67_learnable_seblock/model/pascal_voc_epoch_9.pth" \
    --current_epoch=10
    
TYPE=ms
THR=0.35

CUDA_VISIBLE_DEVICES=3 python3 ./scripts/test_l2g_voc_RM.py \
    --img_dir=./data/voc12/JPEGImages/ \
    --test_list=./data/voc12/train_cls.txt \
    --arch=vgg \
    --batch_size=1 \
    --input_size=224 \
    --num_classes=20 \
    --restore_from=./runs/${EXP}/model/pascal_voc_epoch_12.pth \
    --save_dir=./runs/${EXP}/${TYPE}/attention/ \
    --multi_scale \
    --random_walk_dir_aff=./runs/${EXP}/cam_npy_aff/ \
    --random_walk_dir_irn=./runs/${EXP}/cam_npy_irn/ \
    --cam_npy=./runs/${EXP}/cam_npy/ \
    --cam_png=./runs/${EXP}/cam_png/ \
    --thr=${THR} 
    
#python3 evaluate.py --experiment_name ${EXP}