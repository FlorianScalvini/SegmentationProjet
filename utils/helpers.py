import os
import torch
import torch.nn as nn
import numpy as np
import math
import PIL

def create_path(path):
    if not os.path.exists(path):
            os.makedirs(path)
            print("The path is created")
    else:
        print("The path was already created")

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask