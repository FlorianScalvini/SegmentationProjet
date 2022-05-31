import json
import torch.utils.data
from torch.utils.data import Dataset
from transform import *
import inspect
import utils.loss as loss
import models
import transform
import Dataset



def _transform(transform_dict):
    transform_list = []
    for key in transform_dict:
        value = transform_dict[key]
        if value == False:
            continue
        if value is True or value is None:
            transf = getattr(transform, key)()
        elif isinstance(value, dict):
            transf = getattr(transform, key)(**value)
        else:
            transf = getattr(transform, key)(value)
        transform_list.append(transf)
    return transform_list


class ConfigParser:
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



    def model(self):
        net = getattr(models, self.config['arch']['type'])
        if 'backbone' in inspect.getfullargspec(net.__init__)[0]:
            try:
                net_backbone = getattr(models.backbone, self.config['arch']['backbone']['type'])
                b_kwargs = self.config['arch']['backbone']['args']
            except:
                raise ValueError("Missing backbone value or incorrect config.json")
        else:
            net_backbone = None
            b_kwargs = None
        return net, self.config['arch']['args'], net_backbone,  b_kwargs

    def loss_config(self):
        lss = getattr(loss, self.config['loss']['type'])
        return lss, self.config['loss']['args'], self.config['loss']['coef']

    def optimizer_config(self):
        optim = getattr(torch.optim, self.config['optimizer']['type'])
        return optim, self.config['optimizer']['args']

    def scheduler(self):
        schl = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['type'])
        return schl, self.config['lr_scheduler']['args']

    def train_loader(self):
        loader_config = self.config['train_loader']['args']
        args_dataset = self.config['train_loader']["dataset"]["args"]
        args_dataset['transforms'] = _transform(args_dataset["transforms"])
        train_data = getattr(Dataset, self.config['train_loader']["dataset"]['type'])
        return loader_config, train_data, args_dataset

    def val_loader(self):
        loader_config = self.config['val_loader']['args']
        args_dataset = self.config['val_loader']["dataset"]["args"]
        args_dataset['transforms'] = _transform(args_dataset["transforms"])
        train_data = getattr(Dataset, self.config['val_loader']["dataset"]['type'])
        return loader_config, train_data, args_dataset


    def trainer(self):
        dct = self.config['trainer']
        dct["device"] = self.config['global']['device']
        return dct

    def trainer_config(self):
        dict_return = self.config['trainer']
        kwargs_loader, dataset, kwargs_dataset = self.train_loader()
        train_data = dataset(**kwargs_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, **kwargs_loader)
        dict_return["train_loader"] = train_loader
        if self.val:
            kwargs_loader, dataset, kwargs_dataset = self.val_loader()
            val_data = dataset(**kwargs_dataset)
            val_loader = torch.utils.data.DataLoader(dataset=val_data, **kwargs_loader)
            dict_return["val_loader"] = val_loader
        mdl, kwargs, bck_mdl, bck_kwargs = self.model()
        bck_mdl = bck_mdl(**bck_kwargs)
        if 'num_classes' != kwargs.keys():
            kwargs["num_classes"] = train_data.num_classes
        model = mdl(backbone=bck_mdl, **kwargs)
        dict_return["model"] = model
        loss, kwargs, loss_coef = self.loss_config()
        loss = loss(**kwargs)
        dict_return["loss"] = loss
        dict_return["lossCoef"] = loss_coef
        optim, kwargs = self.optimizer_config()
        if 'num_classes' != kwargs.keys():
            kwargs["params"] = model.parameters()
        optim = optim(**kwargs)
        dict_return["optimizer"] = optim
        scheduler, kwargs = self.scheduler()
        scheduler = scheduler(optimizer=optim, **kwargs)
        dict_return["scheduler"] = scheduler
        dict_return["device"] = self.config['global']['device']
        return dict_return


