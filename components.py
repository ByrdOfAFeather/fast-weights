import torch
import torch.nn as nn


class Linear(nn.Module):
	def __init__(self, in_features, out_features):
		super(Linear, self).__init__()
		self.weight = (-1 - 1) * torch.rand([in_features, out_features]) + 1
		# self.bias = (-1 - 1) * torch.rand([1, out_features]) + 1
		self.in_features = in_features
		self.out_features = out_features

	def forward(self, x):
		# return torch.matmul(x, self.weight) + self.bias
		return torch.matmul(x, self.weight)
