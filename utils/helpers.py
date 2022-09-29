import os
import torch
import torch.nn as nn
import numpy as np
import math
import PIL
import requests
import matplotlib.pyplot as plt


def create_path(path):
    if not os.path.exists(path):
            os.makedirs(path)
            print("The path is created")
    else:
        print("The path was already created")

def colorize_mask(mask, palette):
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def load_from_url(path):
    response = requests.get(path)
    state_dict = response.content
    return state_dict

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def labeltoColor(label, color_map, num_classes, device='cpu'):
    if not isinstance(color_map, dict) or len(color_map) < num_classes:
        raise ValueError("Color map should be a dict or should be equal or greater than the number of classe")
    h, w = label.shape
    color_img = torch.zeros((3, h, w)).to(torch.uint8).to(device)
    for idx in range(num_classes):
        color_img[:, label == idx] = torch.Tensor(torch.FloatTensor([color_map[idx]]).to(device)).transpose(0, 1).to(torch.uint8)
    return color_img