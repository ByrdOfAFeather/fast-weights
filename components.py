import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torch import _VF as F


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
		self.device = device
		self.weight = (-1 - 1) * FastTensor(torch.rand([in_features, out_features]), device=device) + 1
		if bias:
			self.bias = (-1 - 1) * FastTensor(torch.rand([1, out_features]), device=device) + 1
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
		# end_of_weight_idx = idx + self.no_fast_weights
		# if self.bias is not None:
		# 	end_of_weight_idx -= self.out_features
		# 	bias_update = update[end_of_weight_idx: end_of_weight_idx + self.out_features, :].reshape(self.bias.shape)
		# 	self.bias = update_func(self.bias, bias_update)
		# 	end_of_weight_idx += self.out_features
		# weight_update = update[idx:end_of_weight_idx, :].reshape(self.weight.shape)
		# self.weight = update_func(self.weight, weight_update)
		end_of_weight_idx = self.weight.update_weights(update, idx, update_func)
		return end_of_weight_idx

	def reset(self):
		self.weight = self.weight
		self.bias = self.bias

	def forward(self, x):
		ret = torch.matmul(x, self.weight)
		if self.bias is not None:
			return ret + self.bias
		return ret





class RecurrentNet(nn.Module):
	"""
	Currently supports RNN and LSTM
	For RNN support: rnn = RecurrentNet(input_size, hidden_size) or rnn = RNN(input_size, hidden_size)
	For LSTM support: rnn = RecurrentNet(input_size, hidden_size, mode="LSTM") or rnn = LSTM(input_size, hidden_size)
	"""
	def __init__(self, input_size, hidden_size, device="cpu", mode_name="RNN_TANH", bidirectional=False):
		super(RecurrentNet, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.gate_size = None
		self.bidirectional = bidirectional
		self.device = device
		self.mode_name = mode_name
		self.hh_w, self.hi_w, self.no_fast_weights = None, None, None
		self.hh_w_r, self.hi_w_r = None, None

		if self.mode_name == "LSTM":
			self.init_lstm_weights()
			self.mode_func = F.lstm

		else:
			self.init_rnn_weights()
			self.mode_func = F.rnn_tanh

		self.flat_weights = [self.hi_w, self.hh_w]
		self.flatten_weights()

	def init_weights(self, gate_size, hidden_size, input_size):
		"""When called from RNN perspective, gate_size=hidden_size
		"""
		self.hh_w = torch.rand([gate_size, hidden_size], device=self.device).float()
		self.hi_w = torch.rand([gate_size, input_size], device=self.device).float()
		self.no_fast_weights = gate_size * hidden_size + gate_size * input_size
		if self.bidirectional:
			self.hh_w_r = torch.rand([gate_size, hidden_size], device=self.device).float()
			self.hi_w_r = torch.rand([gate_size, input_size], device=self.device).float()
			self.no_fast_weights *= 2

	def init_lstm_weights(self):
		self.gate_size = self.hidden_size * 4
		self.init_weights(self.gate_size, self.hidden_size, self.input_size)

	def init_rnn_weights(self):
		self.init_weights(self.hidden_size, self.hidden_size, self.input_size)

	def flatten_weights(self):
		self.flat_weights = [self.hi_w, self.hh_w]
		if self.bidirectional:
			self.flat_weights.append(self.hi_w_r)
			self.flat_weights.append(self.hh_w_r)
		if self.device == "cuda":
			import torch.backends.cudnn.rnn as rnn
			torch._cudnn_rnn_flatten_weight(
				self.flat_weights, 2,
				self.input_size, rnn.get_cudnn_mode(self.mode_name),
				self.hidden_size, 0, 1,
				False, False)

	def update_weights(self, update, idx, update_func) -> int:
		final_idx = idx + self.no_fast_weights
		end_of_hh_idx = idx + self.hh_w.shape[0] * self.hh_w.shape[1]

		hh_update = update[idx:end_of_hh_idx, :].reshape(self.hh_w.shape)
		hi_update = update[end_of_hh_idx:final_idx, :].reshape(self.hi_w.shape)
		self.hh_w = update_func(self.hh_w, hh_update)
		self.hi_w = update_func(self.hi_w, hi_update)

		if self.bidirectional:
			end_of_hh_r_idx = final_idx + self.hh_w_r.shape[0] * self.hh_w_r.shape[1]
			final_idx = idx + self.no_fast_weights
			hh_r_update = update[final_idx:end_of_hh_r_idx, :].reshape(self.hh_w_r.shape)
			hi_r_update = update[end_of_hh_r_idx: final_idx, :].reshape(self.hi_w_r.shape)
			self.hh_w_r = update_func(self.hh_w_r, hh_r_update)
			self.hi_w_r = update_func(self.hi_w_r, hi_r_update)

		self.flatten_weights()
		return final_idx

	def forward(self, x):
		"""TODO : Implementation that works with PackedSequences
		:param x:
		:return:
		"""
		hx = torch.zeros(1, 1, self.hidden_size, device=self.device)
		out, hid = self.mode_func(x, hx, self.flat_weights, False, 1, 0.0, True, False, False)
		return out, hid


class LSTM(RecurrentNet):
	def __init__(self, input_size, hidden_size, device="cpu", bidirectional=False):
		super(LSTM, self).__init__(input_size, hidden_size, device=device, mode_name="LSTM", bidirectional=bidirectional)

	def forward(self, x):
		is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
		if is_packed:
			inputs, batch_sizes, sorted_indices, unsorted_indices = x
			max_batch_size = int(batch_sizes[0])
		else:
			inputs = x
			batch_sizes, sorted_indices, unsorted_indices = None, None, None
			max_batch_size = inputs.size(1)

		num_dim = 1
		if self.bidirectional: num_dim = 2
		h_z = torch.zeros(num_dim, max_batch_size, self.hidden_size, device=self.device)
		c_z = torch.zeros(num_dim, max_batch_size, self.hidden_size, device=self.device)
		hx = (h_z, c_z)

		if is_packed:
			res = self.mode_func(inputs, hx, self.flat_weights, False, 1, 0.0, True, self.bidirectional, False)
			out, hid = res[0], res[1:]
			output_packed = PackedSequence(out, batch_sizes, sorted_indices, unsorted_indices)
			return output_packed, permute_hidden(hid, unsorted_indices)

		else:
			res = self.mode_func(inputs, hx, self.flat_weights, False,
			                     1, 0.0, True, self.bidirectional, False)
			out, hid = res[0], res[1:]
			return out, hid

#
# test = LSTM(300, 128, bidirectional=False)
# # print(test(torch.rand([2, 4, 1]))[1][0].shape)
# x = test(pack_padded_sequence(torch.rand([100, 50, 300]), np.array([100-i for i in range(50)]), batch_first=False))
# print(x[1][0].shape)
