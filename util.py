import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import cv2
import numpy as np
from skimage import filters
from skimage.color import rgb2gray
from scipy.misc import imread, imresize, imsave, imshow
import png
from skimage.morphology import erosion, square, dilation
from skimage.filters import threshold_mean, threshold_otsu
from PIL import Image, ImageEnhance, ImageFilter
import torch.autograd.variable as Variable

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = imread(filepath)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    shape=img.shape
    img = imresize(img, (128, 128))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = preprocess_img(img)
    return img, shape


def save_img(img, filename, shape):
    
    img = deprocess_img(img)
    img = img.numpy()
    img *= 255
    img = img.clip(0, 255)
    img = np.transpose(img, (1, 2, 0))
    img = imresize(img, (shape[0], shape[1], 1))
    img=rgb2gray(img)
    thresh = threshold_otsu(img)
    img= img > thresh
    img = img.astype(np.uint8)
    img=img*255
    imsave(filename, img)
    print ("Image saved as {}".format(filename))

def preprocess_img(img):
    # [0,255] image to [0,1]
    min = img.min().float()
    max = img.max().float()
    img = torch.FloatTensor(img.size()).copy_(img)
    img.add_(-min).mul_(1.0 / (max - min))

    # RGB to BGR
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [0,1] to [-1,1]
    img = img.mul_(2).add_(-1)

    # check that input is in expected range
    assert img.max() <= 1, 'badly scaled inputs'
    assert img.min() >= -1, "badly scaled inputs"

    return img

def deprocess_img(img):
    # BGR to RGB
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [-1,1] to [0,1]
    img = img.add_(1).div_(2)

    return img
