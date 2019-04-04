from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import numpy as np
from models import G
from util import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='cuhk')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)
# opt.input_nc, opt.output_nc, opt.ngf
netG_state_dict = torch.load(opt.model)
netG = G(opt.input_nc, opt.output_nc, opt.ngf)
netG.load_state_dict(netG_state_dict)

image_dir = "dataset/{}/val_2018/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]


batchsize=2
for image_name in image_filenames:
    img, shape = load_img(image_dir + image_name)
    
    input_x_np = np.zeros((batchsize, 3, 128, 128)).astype(np.float32)
   
    input_x_np[0,:] = np.asarray(img[0])
 
    input= Variable(torch.from_numpy(input_x_np))

    if opt.cuda:
        netG = netG.cuda()
        input = input.cuda()

    out = netG(input)

    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/128x128/2018/cGAN/{}".format(opt.dataset, image_name), shape)
