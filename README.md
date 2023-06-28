
The Official PyTorch code for ["Combination of Suppression and Attention Modules with Online Data Augmentation on L2G for Weakly Supervised Semantic Segmentation"](https://arxiv.org/abs/2204.03206), which is implemented based on the code of [L2G](https://github.com/PengtaoJiang/L2G). 


## Installation
Use the following command to prepare your enviroment.
```
pip install -r requirements.txt
```

Download the PASCAL VOC dataset. 
- [PASCAL VOC 2012 提取碼 cl1e](https://pan.baidu.com/s/1CCR840MJ3Rx7jQ-r1jLX9g)

L2G uses the off-the-shelf saliency maps generated from PoolNet. Download them and move to a folder named **Sal**.
- [Saliency maps for PASCAL VOC 2012](https://drive.google.com/file/d/1ZBLZ3YFw6yDIRWo0Apd4znOozg-Buj4A/view?usp=sharing)  

We use CDA to imporve model perforance. Download high confidence foreground
- [CDA](https://drive.google.com/file/d/1fsql-nXceo4VHidpL3MUGKobz040pANC/view?usp=sharing)

The data folder structure should be like:
```
L2G
├── models
├── scripts
├── utils
├── sem_seg_fg
├── data
│   ├── voc12
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── SegmentationClassAug
│   │   ├── Sal
│   ├── coco14
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── Sal

```
Download the [pretrained model](https://drive.google.com/file/d/15F13LEL5aO45JU-j45PYjzv5KW5bn_Pn/view) 
to initialize the classification network and put it to `./models/`.

## L2G
To train a L2G model on dataset VOC2012, Model weight in `runs/${EXP}/model/pascal_voc_epoch_9.pth`: 
```
cd L2G/
bash train_test_L2G_voc.sh
```

Evaluate the model performance，Result will show on the screen:
```
python3 evaluate.py --experiment_name ${EXP} # in train_test_L2G_voc.sh
```

Generate CAM and pseudo labels from L2G model，Pseudo labels in `/data/voc12/pseudo_seg_labels_{EXP}/`：
```
bash test_l2g_voc.sh # EXP NEED same as train_test_L2G_voc.sh
bash gen_gt_voc.sh # EXP NEED same as train_test_L2G_voc.sh
```
Visualization CAM from L2G model ，Result in `runs/${EXP}/ms/accu_att_zoom`：
```
python3 res.py --experiment_name ${EXP} # in train_test_L2G_voc.sh
```
  
  

## Weakly Supervised Segmentation
The segmentation framework is borrowed from [CLIMS](https://github.com/CVI-SZU/CLIMS/tree/master/segmentation/deeplabv2).

#### Step 1:
Copy `/pseudo_seg_labels` to `deeplabv2/VOCdevkit/VOC2012/pseudo_seg_labels/`

#### Step 2:
Please download pre-trained weights from [Here](https://drive.google.com/drive/folders/1nsXWLoK1w56iC9DE5jwdcqQDX8of4DH5?usp=share_link) and put it into the directory `weights/.`

#### Step 3:
Download datasets [PASCAL VOC 2012](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md).

#### Step 4:
```
bash run_voc12_coco_pretrained.sh
```

## Performance
Methods | mIoU(val) | mIoU (test)  
--- |:---:|:---:
L2G  | 72.1 | 71.7
Proposed Method  | 72.46(+0.36) | 72.81(+1.11)
Segmentation Results on VOC2012 testing dataset at http://host.robots.ox.ac.uk:8080/anonymous/ILSJAD.html

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for NonCommercial use only. Any commercial use should get formal permission first.