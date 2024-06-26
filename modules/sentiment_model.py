from typing import Any
import lightning as L
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from modules.sentiment_dataset import SentimentDataset

class SentimentModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_config = config.model
        self.optim_config = config.optimizer
        self.data_process_config = config.data_process
        self.data_loaders_config = config.dataloaders
        self.tokenizer_config = config.tokenizer
        self.model = object_from_dict(self.model_config)
        self.learning_rate = 0.001

    def forward(self, inputs):
        return self.model(**inputs)
    
    def setup(self, stage = 0):
        self.data = pd.read_csv(**self.data_process_config.dataset)
        trainset, validset = train_test_split(
            self.data,
            **self.data_process_config.splitter, 
            stratify= self.data['labels']
            )
        trainset.reset_index(drop = True, inplace = True)
        validset.reset_index(drop = True, inplace = True)
        self.train_data = SentimentDataset(trainset, self.tokenizer_config)
        self.valid_data = SentimentDataset(validset, self.tokenizer_config)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.train_data.tokenizer.pad_token_id

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_data,
            **self.data_loaders_config.trainloader
            )
        return train_loader
    
    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_data,
            **self.data_loaders_config.validloader
            )
        return valid_loader
        
    def training_step(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        self.log('training_loss', output.loss, prog_bar= True, on_step= False, on_epoch= True)

        return output.loss
    
    def validation_step(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        with torch.no_grad():
            output = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )
        self.log('validation_loss', output.loss, prog_bar= True, on_step= False, on_epoch= True)

        return output.loss

    def configure_optimizers(self):
        optimizer = object_from_dict(self.optim_config, lr = self.learning_rate, params=[x for x in self.model.parameters() if x.requires_grad])
        return optimizer