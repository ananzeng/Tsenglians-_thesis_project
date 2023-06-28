import PIL.Image
import random
import numpy as np
import cv2
import os
from scipy import misc,ndimage

class RandomResizeLong():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        img = img.resize(target_shape, resample=PIL.Image.CUBIC)
        return img

class ResizeShort():
    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, img, isimg=True):

        target_long = self.short_size
        w, h = img.size

        if w < h:
            target_shape = (target_long, int(round(h * target_long / w)))
        else:
            target_shape = (int(round(w * target_long / h)), target_long)
        if isimg:
            img = img.resize(target_shape, resample=PIL.Image.CUBIC)
        else:
            img = img.resize(target_shape, resample=PIL.Image.NEAREST)
        return img 

class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        c, h, w = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((imgarr.shape[0], self.cropsize, self.cropsize), np.float32)
        container[:, cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[:, img_top:img_top+ch, img_left:img_left+cw]

        return container

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def crop_with_box(img, box):
    if len(img.shape) == 3:
        img_cont = np.zeros((max(box[1]-box[0], box[4]-box[5]), max(box[3]-box[2], box[7]-box[6]), img.shape[-1]), dtype=img.dtype)
    else:
        img_cont = np.zeros((max(box[1] - box[0], box[4] - box[5]), max(box[3] - box[2], box[7] - box[6])), dtype=img.dtype)
    img_cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return img_cont


def random_crop(images, cropsize, fills):
    if isinstance(images[0], PIL.Image.Image):
        imgsize = images[0].size[::-1]
    else:
        imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, fills):

        if isinstance(img, PIL.Image.Image):
            img = img.crop((box[6], box[4], box[7], box[5]))
            cont = PIL.Image.new(img.mode, (cropsize, cropsize))
            cont.paste(img, (box[2], box[0]))
            new_images.append(cont)

        else:
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            new_images.append(cont)

    return new_images


class AvgPool2d():

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, img):
        import skimage.measure

        return skimage.measure.block_reduce(img, (self.ksize, self.ksize, 1), np.mean)


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


class RescaleNearest():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, npimg):
        import cv2
        return cv2.resize(npimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)


def bb_IOU(boxA, boxB):
    boxA = [float(aa) for aa in boxA]
    boxB = [float(bb) for bb in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    if xA >= xB or yA >= yB:
        return 0, 0
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    recall = interArea / float(boxAArea)
    # return the intersection over union value
    return iou, recall

def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    for i in range(len(rect)):
        area = rect[i][2]*rect[i][3]
        if large_area < area:
            large_area = area
            target = i

    x = rect[target][0]
    y = rect[target][1]
    w = rect[target][2]
    h = rect[target][3]

    return x, y, w, h


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_arr = np.asarray(img)
        normalized_img = np.empty_like(img_arr, np.float32)

        normalized_img[..., 0] = (img_arr[..., 0] / 255. - self.mean[0]) / self.std[0]
        normalized_img[..., 1] = (img_arr[..., 1] / 255. - self.mean[1]) / self.std[1]
        normalized_img[..., 2] = (img_arr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return normalized_img

################要加的################
def data_aug(origin_img,fg_img,fg_seg, origin_sal, fg_sal, fg_name):
    #print(origin_img.dtype)
    origin_img = cv2.cvtColor(np.array(origin_img), cv2.COLOR_RGB2BGR)
    origin_sal = cv2.cvtColor(np.array(origin_sal), cv2.COLOR_RGB2BGR)
    
    #cv2.imwrite(os.path.join("CDA_Result", fg_name.split(".")[0]+"_ori_img.jpg"), origin_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    #cv2.imwrite(os.path.join("CDA_Result", fg_name.split(".")[0]+"_ori_sal.jpg"), origin_sal, [cv2.IMWRITE_JPEG_QUALITY, 50])

    bg_img=np.empty_like(origin_img)
    bg_sal=np.empty_like(origin_sal) #add
    bg_img[:,:,:]=origin_img[:,:,:]
    bg_sal[:]=origin_sal[:] #add

    #degree=random.randint(-90,90)
    #fg_img=ndimage.rotate(fg_img, degree)
    #fg_seg=ndimage.rotate(fg_seg, degree)
    #fg_sal=ndimage.rotate(fg_sal, degree)

    fg_height=fg_img.shape[0]
    fg_width=fg_img.shape[1]
    fg_aspect_ratio=fg_width/fg_height
    if bg_img.shape[0]*0.7<fg_height:
        fg_height=bg_img.shape[0]*0.7
        fg_width=fg_height*fg_aspect_ratio
    if bg_img.shape[1]*0.7<fg_width:
        fg_width= bg_img.shape[1]*0.7
        fg_height=fg_width/fg_aspect_ratio

    ################0.8-1.2范围内放缩################
    rescale_ratio=random.random()*0.4+0.8
    fg_height=fg_height*rescale_ratio
    fg_width=fg_width*rescale_ratio

    resize_shape=(int(fg_height),int(fg_width))
    fg_img= misc.imresize(fg_img,resize_shape)
    fg_seg= misc.imresize(fg_seg,resize_shape)
    fg_sal= misc.imresize(fg_sal,resize_shape)

    mask=fg_seg!=0 #png貼上的概念，這邊的mask是在用affinitynet的輸出獲得大致的前景輪廓
    mask_not_bool = fg_sal*mask #然後再用大致的前景輪廓*fg_sal獲得更細的前景輪廓
    mask_bool = mask_not_bool>200 #二值化

    row=random.randint(0,bg_img.shape[0]-fg_img.shape[0])
    col=random.randint(0,bg_img.shape[1]-fg_img.shape[1])

    bg_img_modify=bg_img[row:row+resize_shape[0],col:col+resize_shape[1],:]
    for x in range(bg_img_modify.shape[0]):
        for y in range(bg_img_modify.shape[1]):
            if mask_bool[x][y] == True:
                bg_img_modify[x][y] = fg_img[x][y]

    #sal
    bg_sal_modify=bg_sal[row:row+resize_shape[0],col:col+resize_shape[1]]
    for x in range(bg_img_modify.shape[0]):
        for y in range(bg_img_modify.shape[1]):
            if mask_bool[x][y] == True:
                bg_sal_modify[x][y] = fg_sal[x][y]



    aug_img=bg_img
    aug_sal=bg_sal

    #cv2.imwrite(os.path.join("CDA_Result", fg_name.split(".")[0]+"_aug_img.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    #cv2.imwrite(os.path.join("CDA_Result", fg_name.split(".")[0]+"_aug_sal.jpg"), aug_sal, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    #aug_img = PIL.Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    aug_img = PIL.Image.fromarray(aug_img)
    aug_sal = PIL.Image.fromarray(cv2.cvtColor(aug_sal, cv2.COLOR_BGR2RGB)).convert('L')
    
    return aug_img, aug_sal
################要加的################
################要加的################
def cal_overlap(image1, image2):
    pixel_count = 0
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            if image2[x][y] == 255 and image1[x][y][0] == 255:
                pixel_count+=1
    non_zero = cv2.countNonZero(image2)

    return pixel_count/(non_zero+1e-5)

def data_aug_plus(origin_img,fg_img,fg_seg, origin_sal, fg_sal, fg_name):
    #print(origin_img.dtype)
    origin_img = cv2.cvtColor(np.array(origin_img), cv2.COLOR_RGB2BGR)
    origin_sal = cv2.cvtColor(np.array(origin_sal), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(os.path.join("CDA+_Result", fg_name.split(".")[0]+"_ori_img.jpg"), origin_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    cv2.imwrite(os.path.join("CDA+_Result", fg_name.split(".")[0]+"_ori_sal.jpg"), origin_sal, [cv2.IMWRITE_JPEG_QUALITY, 50])

    bg_img=np.empty_like(origin_img)
    bg_sal=np.empty_like(origin_sal) #add
    bg_img[:,:,:]=origin_img[:,:,:]
    bg_sal[:]=origin_sal[:] #add

    #degree=random.randint(-90,90)
    #fg_img=ndimage.rotate(fg_img, degree)
    #fg_seg=ndimage.rotate(fg_seg, degree)
    #fg_sal=ndimage.rotate(fg_sal, degree)

    fg_height=fg_img.shape[0]
    fg_width=fg_img.shape[1]
    fg_aspect_ratio=fg_width/fg_height
    if bg_img.shape[0]*0.7<fg_height:
        fg_height=bg_img.shape[0]*0.7
        fg_width=fg_height*fg_aspect_ratio
    if bg_img.shape[1]*0.7<fg_width:
        fg_width= bg_img.shape[1]*0.7
        fg_height=fg_width/fg_aspect_ratio

    ################0.8-1.2范围内放缩################
    rescale_ratio=random.random()*0.4+0.8
    fg_height=fg_height*rescale_ratio
    fg_width=fg_width*rescale_ratio

    resize_shape=(int(fg_height),int(fg_width))
    fg_img= misc.imresize(fg_img,resize_shape)
    fg_seg= misc.imresize(fg_seg,resize_shape)
    fg_sal= misc.imresize(fg_sal,resize_shape)

    mask=fg_seg!=0 #png貼上的概念，這邊的mask是在用affinitynet的輸出獲得大致的前景輪廓
    mask_not_bool = fg_sal*mask #然後再用大致的前景輪廓*fg_sal獲得更細的前景輪廓
    mask_bool = mask_not_bool>200 #二值化

    row=random.randint(0,bg_img.shape[0]-fg_img.shape[0])
    col=random.randint(0,bg_img.shape[1]-fg_img.shape[1])

    bg_img_modify=bg_img[row:row+resize_shape[0],col:col+resize_shape[1],:]
    for x in range(bg_img_modify.shape[0]):
        for y in range(bg_img_modify.shape[1]):
            if mask_bool[x][y] == True:
                bg_img_modify[x][y] = fg_img[x][y]
            #else: #+
                #bg_img_modify[x][y] = 0 #+
    #bg_img_modify = cv2.cvtColor(bg_img_modify, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(os.path.join("CDA+_Result", fg_name.split(".")[0]+"_aug_img.jpg"), bg_img_modify, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    #sal
    overlap=np.zeros((origin_sal.shape[0], origin_sal.shape[1])) #add
    bg_sal_modify=bg_sal[row:row+resize_shape[0],col:col+resize_shape[1]]
    for x in range(bg_img_modify.shape[0]):
        for y in range(bg_img_modify.shape[1]):
            if mask_bool[x][y] == True:
                bg_sal_modify[x][y] = fg_sal[x][y]
                overlap[x+row][y+col] = 255

    ret, origin_sal_th = cv2.threshold(origin_sal, 200, 255, cv2.THRESH_BINARY)
    miou = cal_overlap(origin_sal_th, overlap)

        
    aug_img=bg_img
    aug_sal=bg_sal

    cv2.imwrite(os.path.join("CDA+_Result", fg_name.split(".")[0]+"_aug_img.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    cv2.imwrite(os.path.join("CDA+_Result", fg_name.split(".")[0]+"_aug_sal.jpg"), aug_sal, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    #aug_img = PIL.Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    aug_img = PIL.Image.fromarray(aug_img)
    aug_sal = PIL.Image.fromarray(cv2.cvtColor(aug_sal, cv2.COLOR_BGR2RGB)).convert('L')
    
    return aug_img, aug_sal, miou
################要加的################