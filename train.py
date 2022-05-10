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
from transform import *
from utils.DataPrefetcher import DataPrefetcher
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
# params of training
    parser.add_argument("--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument('--iters', dest='iters', help='iters for training',type=int, default=None)
    parser.add_argument('--learning_rate', dest='learning_rate', help='Learning rate',type=float, default=None)
    parser.add_argument('--save_interval', dest='save_interval',
                        help='How many iters to save a model snapshot once during training.', type=int, default=1000)
    parser.add_argument('--resume_model',dest='resume_model', help='The path of resume model',type=str, default=None)
    parser.add_argument('--save_dir', dest='save_dir', help='The directory for saving the model snapshot', type=str,
                        default='./output')
    parser.add_argument('--keep_checkpoint_max', dest='keep_checkpoint_max',
                         help='Maximum number of checkpoints to save', type=int, default
                        =5)
    parser.add_argument('--num_workers',dest='num_workers', help='Num workers for data loader',type=int, default=0)
    parser.add_argument('--do_eval', dest='do_eval', help='Eval while training', action='store_true')
    parser.add_argument('--log_iters', dest='log_iters', help='Display logging information at every log_iters',
                         default=10, type=int)
    parser.add_argument('--use_vdl', dest='use_vdl', help='Whether to record the data to VisualDL during training',
                        action='store_true')
    parser.add_argument('--seed', dest='seed', help='Set the random seed during training.', default=None, type=int)
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "fp16"],
                         help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32',"
                              " the training is normal.")
    parser.add_argument("--amp_level", default="O1", type=str, choices=["O1", "O2"],
        help="Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, "
             "the input data type of each operator will be casted by white_list and black_list; O2 represent Pure "
             "fp16, all operators parameters and input data will be casted to fp16, except operators in black_list,"
             " don’t support fp16 kernel and batchnorm. Default is O1(amp)")
    parser.add_argument('--device', dest='device', help='Device place to be set, which can be GPU, XPU, NPU, CPU',
                        default='gpu', type=str)

    return parser.parse_args()





def loss_computation(logits_list, labels, loss):
    loss_coef = loss['coef']
    lossFunct = loss['coef']
    len_logits = len(logits_list)
    if len_logits != len(loss_coef):
        raise ValueError("Different number of logits than loss coef")
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_list.append(lossFunct(logits, labels) * loss_coef[i])
    return loss_list


class Trainer():
    def __init__(self, model, loss, config, train_loader, val_loader=None, train_logger=None, prefetch=True, precision="fp32"):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.start_epoch = 1
        self.improved = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = 100
        # SETTING THE DEVICE
        self.device, _ = self._get_available_devices(self.config['n_gpu'])
        self.model.to(self.device)
        self.scaler = None
        self.optimizer = torch.optim.Adam()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        self.total_loss = 0.
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

        if self.device ==  torch.device('cpu'): prefetch = False
        if precision == 'fp16':
            print('use AMP to train.')
            self.scaler = torch.cuda.amp.GradScaler()
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)
        torch.backends.cudnn.benchmark = True

        avg_loss = 0.0
        avg_loss_list = []
        best_mean_iou = -1.0
        best_model_iter = -1

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()
        tic = time.time()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.optimizer.zero_grad()
            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = loss_computation(logits_list=output, labels=target, loss=self.loss)
                    loss = loss.sum()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer=self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = loss_computation(logits_list=output, labels=target, loss=self.loss)
                loss = loss.sum()
                loss.backward()
                self.optimizer.step()

            preds = torch.argmax(output, dim=1,keepdim=True)
            seg_metrics = eval_metrics(output, target, self.num_classes)
            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                epoch, self.total_loss.average,
                pixAcc, mIoU,
                self.batch_time.average, self.data_time.average))

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
               **seg_metrics}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):

            results = self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)

                # LOGGING INFO
                self.logger.info(f'\n         ## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min':
                        self.improved = (log[self.mnt_metric] < self.mnt_best)
                    else:
                        self.improved = (log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break

                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stoped')
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _reset_metrics(self):
        self.total_loss = 0.
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0


    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")


    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')


def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5,
          test_config=None,
          precision='fp32',
          amp_level='O1',
          profiler_options=None,
          to_static_training=False):


    model.train()
    start_iter = 0
    print(model)
    device = 'gpu'


    train_dataset = cityscrape.Cityscapes(root="/media/ubuntu/DATA/Database/leftImg8bit_trainvaltest/",
                                          mode='train', transforms=[Resize((224,244)), Normalize(32,32)])
    val_dataset = cityscrape.Cityscapes(root="/media/ubuntu/DATA/Database/leftImg8bit_trainvaltest/",
                                          mode='val', transforms=[Resize((224,244)), Normalize(32,32)])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=2, num_workers=4, drop_last=True)
    Trainer(model=model, loss=loss, config=config, train_logger=train_loader, val_loader=val_loader, prefetch=prefetch, precision=precision)