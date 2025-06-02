import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import time
import imageio
import cv2

from model.OCINet_models import OCINet
from data import test_dataset

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path = './dataset/Rail/1130/'

model = OCINet()
model.load_state_dict(torch.load('./models/OCINet.pth',map_location={'cuda:2':'cuda:0'}))
model.cuda()
model.eval()

test_datasets = ['965','165']

for dataset in test_datasets:
    save_path = './models/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/img/'
    print(dataset)
    gt_root = dataset_path + dataset + '/gt/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, s2, s3, s4, res_sig, s2_sig, s3_sig, s4_sig = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res1 = np.uint8(res*255)
        ret, binary = cv2.threshold(res1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(save_path+name[0:-4]+'.jpg', binary)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('FPS {:.5f}'.format(test_loader.size / time_sum))
