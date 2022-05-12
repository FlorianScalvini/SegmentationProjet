import json
import argparse
import random
import time
from tqdm import *
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, rotate, InterpolationMode
import numpy as np
from torch.cuda import amp
from torch.utils.data import DataLoader
from Dataset import cityscrape
from datetime import *
from transform import *
from utils.DataPrefetcher import DataPrefetcher
import os
import logging
from val import evaluate
import models
import transform

def get_instance(module, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config['type'])(*config['args'])

def _transform(transform_dict):
    transform_list = []
    for key, value in transform_dict:
        if value is None or value == False:
            transf = getattr(transform, key)()
        elif isinstance(value, dict):
            transf = getattr(transform, key)(*value)
        else:
            transf = getattr(transform, key)(value)
        transform_list.append(transf)
        return transform_list

class ConfigParser():
    def __init__(self, path):
        try:
            self.config = json.load(open(path))
        except:
            raise ValueError("Wrong path or config file")
        self.name = self.config['name']
        configModel =  self.config['name']['type']
        return

    def model(self):
        net = getattr(models, self.config['arch']['type'])
        return net, self.config['arch']['type']

    def optimizer(self):
        optim = getattr(torch.optim, self.config['optimizer']['type'])
        return optim, self.config['optimizer']['type']

    def scheduler(self):
        schl = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['type'])
        return schl, self.config['lr_scheduler']['type']

    def trainLoader(self):
        args = self.config['train_loader']['args']
        args['transform'] = _transform(self.config["train_loader"]["transform"])
        train_loader = getattr(Dataset, self.config['train_loader']['type'])(self.config['name']['args'])
        return train_loader, args

    def validLoader(self):
        args = self.config['val_loader']['args']
        args['transform'] = _transform(self.config["val_loader"]["transform"])
        val_loader = getattr(Dataset, self.config['val_loader']['type'])(self.config['name']['args'])
        return val_loader, args

    def




