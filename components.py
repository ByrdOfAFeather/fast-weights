import torch
import torch.nn as nn
from torch import _VF as F


class Linear(nn.Module):
	def __init__(self, in_features, out_features, bias=False, device="cpu"):
		super(Linear, self).__init__()
		self.device = device
		self.weight = (-1 - 1) * torch.rand([in_features, out_features], device=self.device) + 1
		if bias:
			self.bias = (-1 - 1) * torch.rand([1, out_features], device=self.device) + 1
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
		end_of_weight_idx = idx + self.no_fast_weights
		if self.bias is not None:
			end_of_weight_idx -= self.out_features
			bias_update = update[end_of_weight_idx: end_of_weight_idx + self.out_features, :].reshape(self.bias.shape)
			self.bias = update_func(self.bias, bias_update)
		weight_update = update[idx:end_of_weight_idx, :].reshape(self.weight.shape)
		self.weight = update_func(self.weight, weight_update)

	def reset(self):
		self.weight = self.weight
		self.bias = self.bias

	def forward(self, x):
		ret = torch.matmul(x, self.weight)
		if self.bias is not None:
			return ret + self.bias
		return ret


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, device="cpu", gate_size=None):
		super(RNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.gate_size = gate_size
		self.device = device
		if gate_size:
			self.hh_w = torch.rand([gate_size, hidden_size], device=self.device).float()
			self.hi_w = torch.rand([gate_size, input_size], device=self.device).float()
			self.no_fast_weights = gate_size * hidden_size + gate_size * input_size
			self.mode_func = F.lstm
			self.mode_name = "LSTM"
		else:
			self.hh_w = torch.rand([hidden_size, hidden_size], device=self.device).float()
			self.hi_w = torch.rand([hidden_size, input_size], device=self.device).float()
			self.no_fast_weights = hidden_size * hidden_size + hidden_size * input_size
			self.mode_func = F.rnn_tanh
			self.mode_name = "RNN_TANH"
		self.flat_weights = [self.hi_w, self.hh_w]
		self.flatten_weights()

	def flatten_weights(self):
		self.flat_weights = [self.hi_w, self.hh_w]
		if self.device == "cuda":
			import torch.backends.cudnn.rnn as rnn
			torch._cudnn_rnn_flatten_weight(
				self.flat_weights, 2,
				self.input_size, rnn.get_cudnn_mode(self.mode_name),
				self.hidden_size, 0, 1,
				False, False)

	def update_weights(self, update, idx, update_func):
		if self.gate_size:
			end_of_weight_idx = idx + self.no_fast_weights
			end_of_hh_idx = idx + self.gate_size * self.hidden_size
		else:
			end_of_weight_idx = idx + self.no_fast_weights
			end_of_hh_idx = idx + self.hidden_size * self.hidden_size

		hh_update = update[idx:end_of_hh_idx, :].reshape(self.hh_w.shape)
		hi_update = update[end_of_hh_idx:end_of_weight_idx, :].reshape(self.hi_w.shape)
		self.hh_w = update_func(self.hh_w, hh_update)
		self.hi_w = update_func(self.hi_w, hi_update)
		self.flatten_weights()

	def forward(self, x):
		hx = torch.zeros(1, 1, self.hidden_size, device=self.device)
		out, hid = self.mode_func(x, hx, self.flat_weights, False, 1, 0.0, True, False, False)
		return out, hid


class LSTM(RNN):
	def __init__(self, input_size, hidden_size, device="cpu"):
		self.mode_func = F.lstm
		self.mode_name = "LSTM"
		super(LSTM, self).__init__(input_size, hidden_size, device=device, gate_size=4 * hidden_size)

	def forward(self, x):
		h_z = torch.zeros(1, 1, self.hidden_size, device=self.device)
		c_z = torch.zeros(1, 1, self.hidden_size, device=self.device)
		hx = (h_z, c_z)
		res = self.mode_func(x, hx, self.flat_weights, False, 1, 0.0, True, False, False)
		out, hid = res[0], res[1:]
		return out, hid