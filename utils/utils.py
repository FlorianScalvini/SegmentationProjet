import numpy as np
import PIL.Image
import cv2
import os
import torch
import torchvision.transforms as T

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def labeltoColor(label, color_map, num_classes, device='cpu'):
    if not isinstance(color_map, dict) or len(color_map) < num_classes:
        raise ValueError("Color map should be a dict or should be equal or greater than the number of classe")
    h, w = label.shape
    color_img = torch.zeros((3, h, w)).to(torch.uint8).to(device)
    for idx in range(num_classes):
        color_img[:, label == idx] = torch.Tensor(torch.ByteTensor([color_map[idx]]).to(device)).transpose(0, 1)
    return color_img


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))