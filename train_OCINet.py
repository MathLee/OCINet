import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.OCINet_models import OCINet
from data import get_loader
from data import test_dataset
from utils import clip_gradient, adjust_lr
from scipy import misc
import imageio
import time
import cv2
from PIL import Image

import pytorch_iou

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)

torch.cuda.set_device(0)

def run():
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        model.train()
        for i, pack in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()

            s1, s2, s3, s4, s1_sig, s2_sig, s3_sig, s4_sig = model(images)

            loss = CE(s1, gts)+IOU(s1_sig, gts)+ (CE(s2, gts)+IOU(s2_sig, gts)) \
                   + (CE(s3, gts)+IOU(s3_sig, gts))/2 + (CE(s4, gts)+IOU(s4_sig, gts))/4

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if i % 100 == 0 or i == total_step:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step,
                               opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data))
        save_path = 'models/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'OCINet.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)


print("Let's go!")
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=70, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()
# build train_models
model = OCINet()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
datasets = 'Rail/2086'
image_root = './dataset/' + datasets + '/img/'
gt_root = './dataset/' + datasets + '/gt/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

print('Learning Rate: {}'.format(opt.lr))

run()


