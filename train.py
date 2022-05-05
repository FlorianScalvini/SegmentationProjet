
import random
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, rotate, InterpolationMode
import numpy as np
import torch


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
    if precision == 'fp16':
        print('use AMP to train. AMP level = {}'.format(amp_level))
    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()

