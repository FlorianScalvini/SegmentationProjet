import config
from train import Trainer
import models



if __name__ == "__main__":
    config = config.ConfigParser("/home/ubuntu/PycharmProjects/SegmentationProjet/config.json")
    trainer_config = config.trainer_config()
    train = Trainer(**trainer_config)
    train.train()
    print("END")


