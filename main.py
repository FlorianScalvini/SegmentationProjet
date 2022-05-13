import torch.utils.data
import torchvision.models as models
import torch.nn as nn
import numpy as np
import config
import models


config = config.ConfigParser("/home/ubuntu/PycharmProjects/SegmentationProjet/config.json")
kwargs_loader, dataset, kwargs_dataset = config.train_loader()
train_data = dataset(**kwargs_dataset)
train_loader = torch.utils.data.DataLoader(dataset=train_data, **kwargs_loader)

val_loader = None
if config.val:
    kwargs_loader, dataset, kwargs_dataset = config.val_loader()
    val_data = dataset(**kwargs_dataset)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, **kwargs_loader)

mdl, kwargs = config.model()
if 'num_classes' == kwargs.keys():
    model = mdl(**kwargs)
else:
    models = mdl(num_classes=train_data.num_classes, **kwargs)

print("END")