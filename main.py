import config
from train import Trainer
import models


config = config.ConfigParser("/home/ubuntu/PycharmProjects/SegmentationProjet/config.json")
trainer_config = config.trainer_config()
train = Trainer(**trainer_config)
train.train()
print("END")