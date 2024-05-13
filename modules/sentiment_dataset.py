from iglovikov_helper_functions.config_parsing.utils import object_from_dict
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer_config):
        self.data = data
        self.texts = self.data['text']
        self.labels = self.data['labels']
        self.config = tokenizer_config
        self.tokenizer = object_from_dict(self.config.tokenizer_loader)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_data = self.texts[idx]
        label_data = torch.tensor(self.labels[idx])
        inputs = self.tokenizer(text_data, **self.config.tokenizer_parameters)

        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label_data.squeeze(0)