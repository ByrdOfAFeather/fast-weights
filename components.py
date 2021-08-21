import torch
import torch.nn as nn


class Linear(nn.Module):
	def __init__(self, in_features, out_features, bias=False):
		super(Linear, self).__init__()
		self.weight = (-1 - 1) * torch.rand([in_features, out_features]) + 1
		if bias:
			self.bias = (-1-1) * torch.rand([1, out_features]) + 1
		else:
			self.bias = None

		self.in_features = in_features
		self.out_features = out_features

		# Used in Brute Force
		self.no_fast_weights = in_features * out_features if not bias else (in_features * out_features) + out_features

		# Used in FROM/TO Architecture
		self.no_from = in_features
		self.no_to = out_features

	def update_weights(self, update, idx, update_func):
		weight_idx = idx + self.no_fast_weights
		if self.bias is not None:
			weight_idx -= self.out_features
			bias_update = update[weight_idx: weight_idx + self.out_features, :].reshape(self.bias.shape)
			self.bias = update_func(self.bias, bias_update)
		weight_update = update[idx:weight_idx, :].reshape(self.weight.shape)
		self.weight = update_func(self.weight, weight_update)

	def forward(self, x):
		ret = torch.matmul(x, self.weight)
		if self.bias is not None:
			return ret + self.bias
		return ret


class RNN(nn.RNN):
	def __init__(self, *args, **kwargs):
		super(RNN, self).__init__(*args, **kwargs)