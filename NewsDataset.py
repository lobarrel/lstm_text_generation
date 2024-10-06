import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        label = torch.tensor(row["Class Index"])
        text = row["Title"] + " " + row["Description"]
        return label, text

    def __len__(self):
        return len(self.dataframe)