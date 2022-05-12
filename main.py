import torchvision.models as models
import torch.nn as nn
import numpy as np
import config
import models


config = config.ConfigParser("/Users/florianscalvini/PycharmProjects/SegmentationProjet/config.json")
func, args = config.trainLoader()
train_loader = func(*args)
val_loader = None
if config.val:
    func, args = config.validLoader()
    val_loader = func(*args)
func, args = config.model()
if 'num_classes' == args.keys():
    model = func(*args)
else:
    models = func(num_classes=train_loader.database.num_classes, *args)
