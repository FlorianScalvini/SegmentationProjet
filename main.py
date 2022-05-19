import torch.utils.data
import torchvision.models as models
import torch.nn as nn
import numpy as np
import config
from train import Trainer
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

mdl, kwargs, bck_mdl, bck_kwargs = config.model()
bck_mdl = bck_mdl(**bck_kwargs)
if 'num_classes' != kwargs.keys():
    kwargs["num_classes"] = train_data.num_classes
model = mdl(backbone=bck_mdl, **kwargs)

loss, kwargs = config.loss()
loss = loss(**kwargs)

optim, kwargs = config.optimizer()
if 'num_classes' != kwargs.keys():
    kwargs["params"] = model.parameters()
optim = optim(**kwargs)

scheduler, kwargs = config.scheduler()
scheduler = scheduler(optimizer=optim, **kwargs)

global_config = config.trainer()
train = Trainer(model=model, loss=loss, global_config=global_config, train_loader=train_loader, val_loader=val_loader)
print("END")