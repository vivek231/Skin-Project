# Code written and maintained by Vivek Kumar Singh.
# Universitat Rovira I Virgili, Tarragona Spain
# Date: 01/June/2019
from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
from FCA import FCANet, CAM_Module, non_bottleneck_1d

# For input size input_nc x 128 x 128
class G(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(G, self).__init__()
        self.size1=nn.AvgPool2d(1, stride=8)
        self.size3=nn.AvgPool2d(1, stride=4)
        self.size4=nn.AvgPool2d(1, stride=2)
        self.convinput= nn.Conv2d(3, ngf, kernel_size=3,padding=1, bias=False)
        self.factor_in=FCANet(ngf,ngf)
        self.conva=nn.Conv2d(3, ngf, kernel_size=3,padding=1, bias=False)
        self.factor_a= FCANet(ngf,ngf)
        self.convc=nn.Conv2d(3, ngf, kernel_size=3,padding=1, bias=False)
        self.factor_c= FCANet(ngf,ngf)
        self.convd=nn.Conv2d(3, ngf, kernel_size=3,padding=1, bias=False)
        self.factor_d=FCANet(ngf,ngf)

        self.conv1 = nn.Conv2d(ngf, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.c2 = FCANet(ngf*2,ngf*2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.c3 = FCANet(ngf*4,ngf*4)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.c4 = FCANet(ngf*8,ngf*8)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.c5 = FCANet(ngf*8,ngf*8)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.c6 = FCANet(ngf*8,ngf*8)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.c7 = FCANet(ngf*8,ngf*8)
       
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2 , 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf*4, ngf, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf*2, output_nc, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

    def forward(self, input):

        a=self.size1(input)
        c=self.size3(input)
        d=self.size4(input)
        input= self.convinput(input)
        input = self.factor_in(input)
        a = self.conva(a) 
        a = self.factor_a(a)
        a=nn.functional.interpolate(a, size=128, mode='bilinear', align_corners=True)    
        c = self.convc(c)
        c = self.factor_c(c)
        c=nn.functional.interpolate(c, size=128, mode='bilinear', align_corners=True)    
        d = self.convd(d)
        d = self.factor_d(d)
        d=nn.functional.interpolate(d, size=128, mode='bilinear', align_corners=True)
# ================ Aggregation of all the multiscale features============
        x=(input+a+c+d)
        e1 = self.conv1(x)
        e2 = self.batch_norm2(self.conv2(self.leaky_relu(e1)))
        e2 = self.c2(e2)
        e3 = self.batch_norm4(self.conv3(self.leaky_relu(e2)))
        e3 = self.c3(e3)
        e4 = self.batch_norm8(self.conv4(self.leaky_relu(e3)))
        e4 = self.c4(e4)
        e5 = self.batch_norm8(self.conv5(self.leaky_relu(e4)))
        e5 = self.c5(e5)
        e6 = self.batch_norm8(self.conv6(self.leaky_relu(e5)))
        e6 = self.c6(e6)
        e7 = self.conv7(self.leaky_relu(e6))
        e7 = self.c7(e7)
        print ("================================")

        # img=e3[0,:3,:,:].cpu().data.numpy()
        # img=np.transpose(img,(1,2,0))
        # cv2.imshow('e3',img)
        # cv2.waitKey(0)

        # Decoder

        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(e7))))
        d1 = torch.cat((d1_, e6), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1))))
        d2 = torch.cat((d2_, e5), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2))))
        d3 = torch.cat((d3_, e4), 1)
        d4_ = self.batch_norm4(self.dconv4(self.relu(d3)))
        d4 = torch.cat((d4_, e3), 1)
        d5_ = self.batch_norm2(self.dconv5(self.relu(d4)))
        d5 = torch.cat((d5_, e2), 1)
        d6_ = self.batch_norm(self.dconv6(self.relu(d5)))
        d6 = torch.cat((d6_, e1), 1)
        d7 = self.dconv7(self.relu(d6))
        output = self.tanh(d7)
        return output

class D(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(D, self).__init__()
        self.s1=nn.AvgPool2d(1, stride=8)
        self.s3=nn.AvgPool2d(1, stride=4)
        self.s4=nn.AvgPool2d(1, stride=2)
        self.convinput= nn.Conv2d(input_nc + output_nc, ndf, kernel_size=3,padding=1, bias=False)
        self.factor_in = FCANet(ndf,ndf)
        self.conva=nn.Conv2d(input_nc + output_nc, ndf, kernel_size=3,padding=1, bias=False)
        self.factor_a = FCANet(ndf,ndf)
        self.convc=nn.Conv2d(input_nc + output_nc, ndf, kernel_size=3,padding=1, bias=False)
        self.factor_c = FCANet(ndf,ndf)
        self.convd=nn.Conv2d(input_nc + output_nc, ndf, kernel_size=3,padding=1, bias=False)
        self.factor_d = FCANet(ndf,ndf)
        self.conv1 = nn.Conv2d(ndf, ndf, 4, 2, 1)
        self.d1    = FCANet(ndf,ndf)
        self.conv2 = nn.Conv2d(ndf, 1, 4, 2, 1)     

        self.batch_norm2 = nn.BatchNorm2d(ndf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ndf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ndf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        a=self.s1(input)
        c=self.s3(input)
        d=self.s4(input)
        input= self.convinput(input)
        input = self.factor_in(input)
        a = self.conva(a) 
        a = self.factor_a(a)
        a=nn.functional.interpolate(a, size=128, mode='bilinear', align_corners=True)    
        c = self.convc(c)
        c = self.factor_c(c)
        c=nn.functional.interpolate(c, size=128, mode='bilinear', align_corners=True)    
        d = self.convd(d)
        d = self.factor_d(d)
        d=nn.functional.interpolate(d, size=128, mode='bilinear', align_corners=True)
#  ==================  Aggregation of all scales features ===================
        x = (input+a+c+d)
        h1 = self.conv1(x)
        h1 = self.d1(h1)
        h2 = self.conv2(self.leaky_relu(h1))
        output = self.sigmoid(h2)
        return output
