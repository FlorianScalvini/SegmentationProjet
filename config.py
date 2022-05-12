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
import Dataset
import typing

def get_instance(module, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config['type'])(*config['args'])

def _transform(transform_dict):
    transform_list = []
    for key in transform_dict:
        value = transform_dict[key]
        if value == False:
            continue
        if value is True or value is None:
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
        self.name = True if self.config["global"]['name'] else False
        self.train =  True if self.config["global"]['train']  else False
        self.val =  True if self.config["global"]['val']  else False
        self.test =  True if self.config["global"]['test']  else False
        return

    def model(self) -> typing.Tuple[models.BaseModel, dict]:
        net = getattr(models, self.config['arch']['type'])
        return net, self.config['arch']['type']

    def optimizer(self) -> typing.Tuple[torch.optim.Optimizer, dict]:
        optim = getattr(torch.optim, self.config['optimizer']['type'])
        return optim, self.config['optimizer']['type']

    def scheduler(self):
        schl = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['type'])
        return schl, self.config['lr_scheduler']['type']

    def trainLoader(self) -> typing.Tuple[Dataset.Dataset, dict]:
        args = self.config['train_loader']['args']
        args['transform'] = _transform(args["transform"])
        train_loader = getattr(Dataset, self.config['train_loader']['type'])
        return train_loader, args

    def validLoader(self) -> typing.Tuple[Dataset.Dataset, dict]:
        args = self.config['val_loader']['args']
        args['transform'] = _transform(self.config["val_loader"]["transform"])
        val_loader = getattr(Dataset, self.config['val_loader']['type'])
        return val_loader, args

    def trainer(self) -> dict:
        return self.config['trainer']





