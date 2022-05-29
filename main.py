import torch.utils.data
import torchvision.models as models
import torch.nn as nn
import numpy as np
import config
from train import Trainer
import models


config = config.ConfigParser("/home/ubuntu/PycharmProjects/SegmentationProjet/config.json")
trainer_config = config.trainer_config()
train = Trainer(**trainer_config)
print("END")