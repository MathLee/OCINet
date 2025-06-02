import os
import torch
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def batch_pix_accuracy(prediction, target):
    pixel_labeled = (target > 0).sum()
    pixel_correct = ((prediction == target) * (target > 0)).sum()
    pixel_acc = np.divide(pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy())
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy(), pixel_acc


def cal_iou(pred, gt):
    area_inter = np.count_nonzero(pred * gt)
    area_union = np.count_nonzero(pred + gt)
    iou = area_inter * 1.0 / (area_union + np.spacing(1))
    return iou


def batch_intersection_union(prediction, target, num_class):
    # prediction = prediction * (target > 0).long()
    intersection = prediction * (prediction == target).long()
    area_inter = torch.histc(intersection.float(), bins=num_class - 1, max=num_class - 0.9, min=0.1)
    # print(area_inter[0])
    area_pred = torch.histc(prediction.float(), bins=num_class - 1, max=num_class - 0.9, min=0.1)
    area_lab = torch.histc(target.float(), bins=num_class - 1, max=num_class - 0.9, min=0.1)
    area_union = area_pred + area_lab - area_inter
    # print(area_union.float())
    IoU = area_inter.float() / (np.spacing(1) + area_union.float())
    Dice = 2*area_inter.float() / (np.spacing(1) + area_pred.float() + area_lab.float())
    mIoU = IoU.sum() / torch.nonzero(area_lab).size(0)
    mDice = Dice.sum() / torch.nonzero(area_lab).size(0)

    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return mIoU.cpu().numpy(), mDice.cpu().numpy()


if __name__ == '__main__':
    label_dir_165 = '/home/lgy/lgy/20240314CANet_NRSD_1/dataset/Rail/1130/165/gt/'
    label_dir_965 = '/home/lgy/lgy/20240314CANet_NRSD_1/dataset/Rail/1130/965/gt/'

    file_dir_165 = '/home/lgy/lgy/20240314CANet_NRSD_1/models/165/'
    file_dir_965 = '/home/lgy/lgy/20240314CANet_NRSD_1/models/965/'
    curImgDir = file_dir_165
    num_test = 0
    avg_pa_165 = 0
    avg_mIou_165 = 0
    avg_mDice_165 = 0

    for num_test, img in enumerate(os.listdir(curImgDir)):
        image_path = os.path.join(curImgDir, img)
        label_path = os.path.join(label_dir_165, img)

        img = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)  # - 1 # from -1 to 149
        label = label / 255
        image = img / 255

        # iou = cal_iou(image, label)

        image = torch.from_numpy(image).cuda().long()
        true_masks = torch.from_numpy(label).cuda().long()

        pa = batch_pix_accuracy(image, true_masks)
        iou, dice = batch_intersection_union(image, true_masks, 2)
        # iou = cal_iou(image, true_masks)
        avg_pa_165 = avg_pa_165 + pa[2]
        avg_mIou_165 = avg_mIou_165 + iou
        avg_mDice_165 =  avg_mDice_165 + dice

    avg_pa_165 = avg_pa_165 / (num_test + 1)
    avg_mIou_165 = avg_mIou_165 / (num_test + 1)
    avg_mDice_165 =  avg_mDice_165 / (num_test + 1)

    curImgDir = file_dir_965
    num_test = 0
    avg_pa_965 = 0
    avg_mIou_965 = 0
    avg_mDice_965 = 0

    for num_test, img in enumerate(os.listdir(curImgDir)):
        image_path = os.path.join(curImgDir, img)
        label_path = os.path.join(label_dir_965, img)

        img = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)  # - 1 # from -1 to 149
        label = label / 255
        image = img / 255

        image = torch.from_numpy(image).cuda().long()
        true_masks = torch.from_numpy(label).cuda().long()

        pa = batch_pix_accuracy(image, true_masks)
        iou, dice = batch_intersection_union(image, true_masks, 2)
        avg_pa_965 = avg_pa_965 + pa[2]
        avg_mIou_965 = avg_mIou_965 + iou
        avg_mDice_965 = avg_mDice_965 + dice

    avg_pa_965 = avg_pa_965 / (num_test + 1)
    avg_mIou_965 = avg_mIou_965 / (num_test + 1)
    avg_mDice_965 = avg_mDice_965 / (num_test + 1)

    print('Dataset 965: PA:{:.3f}, mIoU:{:.3f}, mDice:{:.3f}; Dataset 165: PA:{:.3f}, mIoU:{:.3f}, mDice:{:.3f}'.format(avg_pa_965,
                                                                                                              avg_mIou_965,
                                                                                                              avg_mDice_965,
                                                                                                              avg_pa_165,
                                                                                                              avg_mIou_165,
                                                                                                              avg_mDice_165))

