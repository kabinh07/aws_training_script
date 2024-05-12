import argparse
from pathlib import Path
import yaml
from addict import Dict as Addict
from modules.sentiment_model import SentimentModel
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
import os
import torch
from lightning.pytorch.tuner.tuning import Tuner

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sentiment():
    def __init__(self, config):
        self.config = config
        if self.config.load_last_checkpoint:
            if os.path.exists(self.config.callbacks.model_checkpoint.dirpath+"/last_epoch_model.ckpt"):
                self.model = SentimentModel.load_from_checkpoint(
                    self.config.callbacks.model_checkpoint.dirpath+"/last_epoch_model.ckpt", 
                    map_location = device,
                    config = self.config
                    )
                # os.remove(self.config.callbacks.model_checkpoint.dirpath+"/last.ckpt")
            else:
                self.model = SentimentModel(self.config)
        else:
            self.model = SentimentModel(self.config)
        self.logger = object_from_dict(self.config.logger)
        self.early_stopping = object_from_dict(self.config.callbacks.early_stopping)
        self.best_model_checkpoint = object_from_dict(self.config.callbacks.best_model_checkpoint)
        self.model_checkpoint = object_from_dict(self.config.callbacks.model_checkpoint)
        self.trainer = object_from_dict(
            self.config.trainer, 
            logger=self.logger, 
            callbacks=[
                self.early_stopping, 
                self.best_model_checkpoint, 
                self.model_checkpoint
                ])
        self.tuner = Tuner(self.trainer)

    def train(self):
        self.trainer.fit(self.model)

    def find_learning_rate(self):
        self.tuner.lr_find(
            model = self.model,
            **self.config.tuner.lr_finder)

def main():
    args = get_args()
    with open(args.config_path) as f:
        config = Addict(yaml.load(f, Loader=yaml.SafeLoader))

    pipeline = Sentiment(config)
    if config.tuner.enable_tuner:
        pipeline.find_learning_rate()
    pipeline.train()

if __name__ == '__main__':
    main()