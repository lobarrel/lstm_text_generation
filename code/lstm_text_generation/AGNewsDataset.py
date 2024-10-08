import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

# AGNews dataset class definition
class AGNews(Dataset):
    def __init__(self, filename):
        # Read the AG News dataset CSV file
        self.df = pd.read_csv(filename, names=["Class", "Title", "Content"])
        
        # Fill missing values with empty string
        self.df.fillna('', inplace=True)
        
        # Combine Title and Content columns into a single Article column
        self.df['Article'] = self.df['Title'] + " : " + self.df['Content']
        
        # Drop Title and Content columns as they are no longer needed
        self.df.drop(['Title', 'Content'], axis=1, inplace=True)
        
        # Replace special characters with whitespace in the Article column
        self.df['Article'] = self.df['Article'].str.replace(r'\\n|\\|\\r|\\r\\n|\n|"', ' ', regex=True)

    # Method to get a single item from the dataset
    def __getitem__(self, index):
        # Get the text of the article at the given index, converted to lowercase
        text = self.df.loc[index]["Article"].lower()

        return text

    # Method to get the length of the dataset
    def __len__(self):
        # Return the total number of articles in the dataset
        return len(self.df)