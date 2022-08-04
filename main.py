import config
from train import Trainer
import models


def test():
    print("ffffff")

if __name__ == "__main__":
    config = config.ConfigParser("C:\\Users\\Florian\\PycharmProjects\\SegmentationProjet\\config.json")
    trainer_config = config.trainer_config()
    train = Trainer(**trainer_config)
    train.train()
    print("END")


