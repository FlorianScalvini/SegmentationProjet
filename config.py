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

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def _transform(transform_dict):
    transform_list = []
    for key, value in transform_dict:
        transf = get_instance(transform, key, value)
        transform_list.append()
        return

class Config():
    def __init__(self, path):
        try:
            self.config = json.load(open(path))
        except:
            raise ValueError("Wrong path or config file")
        self.name = self.config['name']
        configModel =  self.config['name']['type']
        self.models = get_instance(models,self.config['name']['type'], self.config['name']['args'])
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.transform =
        if self.config["train"] == True:

        else

        return

    def trainer(self):
        self.transform =
        self.train_loader = get_instance(Dataset,self.config['train_loader']['type'], self.config['name']['args'])
        return

    def validation(self):
        return

    def test(self):



