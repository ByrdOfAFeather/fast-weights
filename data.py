import torch
import pandas as pd
from torch.utils.data import Dataset


class IrisDataset(Dataset):
	def __init__(self, file_name, sample_amount):
		price_df = pd.read_csv(file_name).sample(frac=sample_amount, random_state=225530)

		x = price_df.iloc[:, 0:4].values
		y = price_df.iloc[:, 4].values

		self.x_train = torch.tensor(x, dtype=torch.float32)
		self.y_train = torch.tensor(y, dtype=torch.long)

	def __len__(self):
		return len(self.y_train)

	def __getitem__(self, idx):
		return self.x_train[idx], self.y_train[idx]