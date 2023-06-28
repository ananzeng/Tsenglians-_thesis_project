from pickle import TRUE
import sys
import os
sys.path.append(os.getcwd())

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import argparse
from utils.LoadData import test_l2g_data_loader
from tqdm import tqdm
import importlib

def get_arguments():
    parser = argparse.ArgumentParser(description='L2G test code')
    parser.add_argument("--save_dir", type=str, default='./runs/exp8/')
    parser.add_argument("--img_dir", type=str, default='./data/VOCdevkit/VOC2012/JPEGImages/')
    parser.add_argument("--test_list", type=str, default='./data/voc12/train_cls.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='voc2012')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--arch", type=str,default='vgg_v0')
    parser.add_argument("--multi_scale", action='store_true', default=False)
    parser.add_argument("--restore_from", type=str, default='./runs/exp7/model/pascal_voc_epoch_14.pth')
    parser.add_argument("--thr", default=0.25, type=float)
    parser.add_argument("--cam_npy", type=str, default='./runs/exp_voc_test/cam_npy/')
    parser.add_argument("--cam_png", type=str, default='./runs/exp_voc_test/cam_png/')
    parser.add_argument("--random_walk_dir_aff", type=str, default='./runs/exp_voc_test/cam_npy_aff/')
    parser.add_argument("--random_walk_dir_irn", type=str, default='./runs/exp_voc_test/cam_npy_irn/')
    parser.add_argument("--DRS_stage", type=str, default='0,0,0,0,0,1,1')   
    return parser.parse_args()

def resize_for_tensors(tensors, size, mode='bilinear', align_corners=False):
    return F.interpolate(tensors, size, mode=mode, align_corners=align_corners)

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

def get_resnet38_model(args):
    model_name = "models.resnet38"
    print(model_name)
    model = getattr(importlib.import_module(model_name), 'Net')(args)

    pretrained_dict = torch.load(args.restore_from)['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return  model

def validate(args):
    print('\nvalidating ... ', flush=True, end='')

    model = get_resnet38_model(args)
    model = model.cuda()
    model.eval()
    val_loader = test_l2g_data_loader(args, args.multi_scale)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.cam_npy is not None: #預設none
        os.makedirs(args.cam_npy, exist_ok=True)
    if args.cam_png is not None: #預設none
        os.makedirs(args.cam_png, exist_ok=True)
    if args.random_walk_dir_aff is not None:
        os.makedirs(args.random_walk_dir_aff, exist_ok=True)   
    if args.random_walk_dir_irn is not None:
        os.makedirs(args.random_walk_dir_irn, exist_ok=True)
        
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img_name, img, label_in = dat
            im_name = img_name[0].split('/')[-1][:-4]
            #print("im_name", im_name)
            if not args.multi_scale:
                img_name, img, label_in = dat
                cv_im = cv2.imread(img_name[0])
                height, width = cv_im.shape[:2]
                _, logits = model(img.cuda())
                cam_map = model.module.get_heatmaps()
                cam_map = F.interpolate(cam_map, (height, width), mode='bicubic', align_corners=False)[0]
                cam_map_np = cam_map.cpu().data.numpy()
            else:
                pred_cam_list = []
                img_name, msf_img_list, label_in = dat 
                cv_im = cv2.imread(img_name[0])
                height, width = cv_im.shape[:2]
                
                strided_size = get_strided_size((height, width), 4) #decrease image size 4 times
                strided_up_size = get_strided_up_size((height, width), 16) #resize image to 16 multiple size 
                cams_list = []
                for m in range(len(msf_img_list)):
                    img = msf_img_list[m]
                    cam_map, logits = model(img.cuda())  
                    cam_map = F.relu(cam_map)
                    cam_map_np = np.squeeze(cam_map, axis=0)
                    cam_map_np = cam_map_np[:-1, :, :]
                    if m % 2 == 1:
                        cam_map_np = cam_map_np.flip(-1)
                    pred_cam_list.append(cam_map_np)
                for m in range(0, len(msf_img_list), 2):
                    cams_list.append(pred_cam_list[m]+pred_cam_list[m+1])
            #+++
            """
                cam_map_np = np.mean(pred_cam_list, axis=0)
            cam_map_np[cam_map_np < 0] = 0
            norm_cam = cam_map_np / (np.max(cam_map_np, (1, 2), keepdims=True) + 1e-8)
            labels = label_in.long().numpy()[0]
            im_name = img_name[0].split('/')[-1][:-4]
            cam_dict = {}
            # cam_dict
            for j in range(args.num_classes):
                if labels[j] > 1e-5:
                    cam_dict[j] = norm_cam[j]
                    out_name = args.save_dir + im_name + '_{}.png'.format(j)
                    cv2.imwrite(out_name, norm_cam[j]*255.0)

            tensor = np.zeros((21, height, width), np.float32)
            for key in cam_dict.keys():
                tensor[key+1] = cam_dict[key]
            tensor[0, :, :] = args.thr
            pred = np.argmax(tensor, axis=0).astype(np.uint8)

            # save cam
            if args.cam_npy is not None:
                np.save(os.path.join(args.cam_npy, im_name + '.npy'), cam_dict)

            if args.cam_png is not None:
                cv2.imwrite(os.path.join(args.cam_png, im_name + '.png'), pred)
            """
            #+++
            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list] #resize each sacle of cam
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0) #[20, strided_size[0], strided_size[1]]

            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :height, :width] #[20, ori_h, ori_w]

            labels = label_in.long().numpy()[0]
            keys = torch.nonzero(torch.from_numpy(labels))[:, 0] #objects index of image(only exsist object) #+++
            
            strided_cams = strided_cams[keys]
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            
            hr_cams = hr_cams[keys] #[len(keys), ori_h, ori_w]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5 #normalization?[0~1]
            
            # save cams
            np.save(os.path.join(args.random_walk_dir_irn, im_name + '.npy'), {"keys": keys, "cam": strided_cams.cpu(), "high_res": hr_cams.cpu().numpy()}) #hr_cam
            
            keys = np.pad(keys + 1, (1, 0), mode='constant') # +1 for background as 0 #因為irn註解
            np.save(os.path.join(args.random_walk_dir_aff, im_name + '.npy'), {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()}) #hr_cam
                
            
            """
            if args.cam_npy is not None:
                np.save(os.path.join(args.cam_npy, im_name + '.npy'), cam_dict)
            if args.cam_png is not None:
                cv2.imwrite(os.path.join(args.cam_png, im_name + '.png'), pred)
            """
if __name__ == '__main__':
    args = get_arguments()
    print(args)
    validate(args)
