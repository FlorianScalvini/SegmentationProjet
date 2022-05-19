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
import inspect
import utils.loss as loss
import models
import transform
import Dataset
import typing
try:
    from typing import GenericMeta  # python 3.6
except ImportError:
    # in 3.7, genericmeta doesn't exist but we don't need it
    class GenericMeta(type): pass



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

def parserToFunc(funct, dict_val):
    return


class ConfigParser():
    def __init__(self, path):
        try:
            self.config = json.load(open(path))
        except:
            raise ValueError("Wrong path or config file")
        try:
            self.train = True if self.config["global"]['train'] else False
            self.val = True if self.config["global"]['val'] else False
            self.test = True if self.config["global"]['test'] else False
        except:
            raise ValueError("Missing information")
        return



    def model(self) -> typing.Tuple[typing.GenericMeta, dict, typing.GenericMeta, dict]:
        net = getattr(models, self.config['arch']['type'])
        if 'backbone' in inspect.getargspec(net.__init__)[0]:
            try:
                net_backbone = getattr(models.backbone, self.config['arch']['backbone']['type'])
                b_kwargs = self.config['arch']['backbone']['args']
            except:
                raise ValueError("Missing backbone value or incorrect config.json")
        else:
            net_backbone = None
            b_kwargs = None
        return net, self.config['arch']['args'], net_backbone,  b_kwargs

    def loss(self) -> typing.Tuple[typing.GenericMeta, dict]:
        lss = getattr(loss, self.config['loss']['type'])
        return lss, self.config['loss']['args']

    def optimizer(self) -> typing.Tuple[typing.GenericMeta, dict]:
        optim = getattr(torch.optim, self.config['optimizer']['type'])
        return optim, self.config['optimizer']['args']

    def scheduler(self) -> typing.Tuple[typing.GenericMeta, dict]:
        schl = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['type'])
        return schl, self.config['lr_scheduler']['args']

    def train_loader(self) -> typing.Tuple[dict, typing.GenericMeta, dict]:
        loader_config = self.config['train_loader']['args']
        args_dataset = self.config['train_loader']["dataset"]["args"]
        args_dataset['transforms'] = _transform(args_dataset["transforms"])
        train_data = getattr(Dataset, self.config['train_loader']["dataset"]['type'])
        return loader_config, train_data, args_dataset

    def val_loader(self) -> typing.Tuple[dict, typing.GenericMeta, dict]:
        loader_config = self.config['val_loader']['args']
        args_dataset = self.config['val_loader']["dataset"]["args"]
        args_dataset['transforms'] = _transform(args_dataset["transforms"])
        train_data = getattr(Dataset, self.config['val_loader']["dataset"]['type'])
        return loader_config, train_data, args_dataset


    def trainer(self) -> dict:
        dct = self.config['trainer']
        dct["device"] = self.config['global']['device']
        return dct




