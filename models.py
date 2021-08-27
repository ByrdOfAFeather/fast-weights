import torch
import torch.nn as nn
import numpy as np
import components as c
from collections import OrderedDict
from toy_example.data import load_episode

torch.manual_seed(225530)
np.random.seed(225530)


class FastNet(nn.Module):
	def __init__(self):
		super(FastNet, self).__init__()
		self.prototype = None
		self.relu = nn.ReLU()
		self.network, self.no_fast_weights = self._build_network()

	@staticmethod
	def _count_weights(weights):
		total_features = 0
		for name, module in weights:
			try:
				total_features += module.no_fast_weights
			except AttributeError:
				pass
		return total_features

	def _build_network(self):
		raise NotImplementedError

	def reset(self):
		self.network, _ = self._build_network()

	def forward(self, x):
		return self.network(x)


class FeedForwardFastNet(FastNet):
	def __init__(self):
		super(FeedForwardFastNet, self).__init__()
		self.no_from = 6
		self.no_to = 4

	def _build_network(self):
		if self.prototype is not None:
			for name, mod in self.prototype:
				mod.reset()

		self.prototype = [
			("linear1", c.Linear(3, 3)),
			("linear2", c.Linear(3, 1)),
		]

		def forward(x):
			cur = x
			for module in self.prototype:
				cur = module[1](cur)
			return cur

		return forward, self._count_weights(self.prototype)


class RNNFastNet(FastNet):
	def __init__(self, input_size, hidden_size, device="cpu"):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.device = device
		super(RNNFastNet, self).__init__()

	def _build_network(self):
		self.prototype = [
			("rnn", c.RNN(self.input_size, self.hidden_size, device=self.device)),
			("linear", c.Linear(self.hidden_size, 1, device=self.device))
		]

		def forward(x):
			final_states, _ = self.prototype[0][1](x)
			preds = torch.zeros([final_states.shape[0], 1], device=self.device)
			for idx, state in enumerate(final_states):
				preds[idx, :] = self.prototype[1][1](state)
			return preds

		return forward, self._count_weights(self.prototype)


class BruteForceUpdater(nn.Module):
	def __init__(self, input_size, fast_net):
		super(BruteForceUpdater, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_size, fast_net.no_fast_weights * 2, bias=False),
			nn.Linear(fast_net.no_fast_weights * 2, fast_net.no_fast_weights, bias=False)
			# nn.Linear(input_size, fast_net.no_fast_weights).cpu()
		)
		self.relu = nn.ReLU()
		self.fast_net = fast_net
		self.update_func = lambda weight, update: 1 / (1 + torch.exp(-10 * (weight + update - .5)))

	def update_weights(self, update):
		prev_weight_idx = 0
		for name, module in self.fast_net.prototype:
			try:
				cur_no_weights = module.no_fast_weights
				module.update_weights(update, prev_weight_idx, self.update_func)
				prev_weight_idx += cur_no_weights
			except AttributeError:
				pass

	def forward(self, x):
		preds = torch.zeros([x.shape[0], 1])
		weight_updates = torch.zeros([x.shape[0], self.fast_net.no_fast_weights, 1])
		for idx, x_i in enumerate(x[:, 0, :]):
			weight_updates[idx, :, :] = self.network(x_i).unsqueeze(1)
			self.update_weights(weight_updates[idx, :, :])
			preds[idx, :] = self.fast_net(x_i)
		return preds


class RNNUpdater(nn.Module):
	def __init__(self, input_size, hidden_size, fast_net, device="cpu"):
		super(RNNUpdater, self).__init__()
		self.device = device
		self.rnn = nn.RNN(input_size, hidden_size, device=device)
		self.linear = nn.Linear(hidden_size, fast_net.no_fast_weights, device=device)

		def net(x):
			_, out = self.rnn(x)
			final_preds = self.linear(out)
			return final_preds

		self.network = net
		self.relu = nn.ReLU()
		self.fast_net = fast_net
		self.update_func = lambda weight, update: 1 / (1 + torch.exp(-10 * (weight + update - .5)))

	def update_weights(self, update):
		prev_weight_idx = 0
		for name, module in self.fast_net.prototype:
			# try:
			cur_no_weights = module.no_fast_weights
			module.update_weights(update, prev_weight_idx, self.update_func)
			prev_weight_idx += cur_no_weights
			# except AttributeError:
			# 	pass

	def forward(self, x):
		weight_update = self.network(x).permute(0, 2, 1).squeeze(0)
		self.update_weights(weight_update)
		preds = self.fast_net(x)
		return preds


class FromToUpdater(nn.Module):
	def __init__(self, input_size, fast_net):
		super(FromToUpdater, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_size, fast_net.no_from + fast_net.no_to)
		)
		self.fast_net = fast_net
		self.update_func = lambda weight, update: 1 / (1 + torch.exp(-10 * (weight + update - .5)))

	def update_weights(self, update):
		prev_from_idx, prev_to_idx = 0, self.fast_net.no_from
		for name, module in self.fast_net.prototype:
			no_from = module.no_from
			no_to = module.no_to
			cur_update = update[prev_to_idx: prev_to_idx + no_to, :] * torch.t(update[prev_from_idx: prev_from_idx + no_from, :])
			module.update_weights(cur_update, 0, self.update_func)
			prev_to_idx += no_to
			prev_from_idx += no_from

	def forward(self, x):
		preds = torch.zeros([x.shape[0], 1])
		for idx, x_i in enumerate(x[:, 0, :]):
			weight_update = self.network(x_i).unsqueeze(1)
			self.update_weights(weight_update)
			preds[idx, :] = self.fast_net(x_i)
		return preds


class RNNBaseline(nn.Module):
	def __init__(self):
		super(RNNBaseline, self).__init__()
		self.rnn = nn.RNN(3, 3)
		self.out_weights = nn.Linear(3, 1)

	def forward(self, x):
		out, h = self.rnn(x)
		out_preds = torch.zeros([x.shape[0], 1])
		for idx, h in enumerate(out):
			out_preds[idx, :] = self.out_weights(h)
		return out_preds