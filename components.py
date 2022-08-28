import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torch import _VF as F, Tensor


def apply_permutation(tensor, permutation, dim=1):
	return tensor.index_select(dim, permutation)


def permute_hidden(hx, permutation):
	if permutation is None:
		return hx
	return apply_permutation(hx, permutation)


class FastTensor:
	def __init__(self, data, *args, **kwargs):
		self.device = kwargs.get("device", "cpu")

		if isinstance(data, torch.Tensor):
			self._tensor = data
		else:
			self._tensor = torch.tensor(data, *args, **kwargs)

		if self.device == "cuda":
			self._tensor = self._tensor.cuda()
		else:
			self._tensor = self._tensor.cpu()

	def update_weights(self, update, idx, update_func) -> int:
		new_idx = idx + np.prod(self._tensor.shape)
		weight_update = update[idx:new_idx, :].reshape(self._tensor.shape)
		self._tensor = update_func(self._tensor, weight_update)
		return new_idx

	def __repr__(self):
		return "data:\n{}".format(self._tensor)

	def __mul__(self, other):
		return FastTensor(self._tensor * other, device=self.device)

	def __rmul__(self, other):
		return FastTensor(other * self._tensor, device=self.device)

	def __add__(self, other):
		return FastTensor(self._tensor + other, device=self.device)

	def __radd__(self, other):
		return FastTensor(other + self._tensor, device=self.device)

	def get_tensor(self):
		return self._tensor

	def __torch_function__(self, func, types, args=(), kwargs=None):
		if kwargs is None:
			kwargs = {}
		args = [a._tensor if hasattr(a, '_tensor') else a for a in args]
		ret = func(*args, **kwargs)
		kwargs["device"] = self.device
		return FastTensor(ret, *args, **kwargs)


class Linear(nn.Module):
	def __init__(self, in_features, out_features, bias=False, device="cpu"):
		super(Linear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.device = device
		self.weight = torch.zeros([self.in_features, self.out_features])
		if bias:
			self.bias = (-1 - 1) * torch.tensor(torch.rand([1, out_features]), device=device) + 1
		else:
			self.bias = None

		self.in_features = in_features
		self.out_features = out_features

		# Used in Brute Force
		self.no_fast_weights = in_features * out_features if not bias else (in_features * out_features) + out_features

		# Used in FROM/TO Architecture
		self.no_from = in_features
		self.no_to = out_features

	def update_weights(self, update, idx, update_func) -> int:
		new_idx = idx + np.prod(self.weight.shape)
		weight_update = update[idx:new_idx, :].reshape(self.weight.shape)
		self.weight = update_func(self.weight, weight_update)
		return new_idx

	def reset(self):
		self.weight = torch.zeros([self.in_features, self.out_features])
		self.bias = torch.zeros([1, self.out_features])

	def forward(self, x):

		ret = torch.matmul(x, self.weight)
		if self.bias is not None:
			return ret + self.bias
		return ret
