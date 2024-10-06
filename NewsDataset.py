import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row["Title"] + " " + row["Description"]
        label = torch.tensor(row["Class Index"])
        return text, label

    def __len__(self):
        return len(self.dataframe)