import torch
import torch.nn as nn
import numpy as np
import components as c
from collections import OrderedDict
from toy_example.data import load_episode

torch.manual_seed(225530)
np.random.seed(225530)
EPISODE_LEN = 300


class FastNetwork(nn.Module):
	def __init__(self):
		super(FastNetwork, self).__init__()
		self.relu = nn.ReLU()
		self.network, self.no_fast_weights = self._build_network()
		self.temp_update = None

	def _build_network(self):
		self.arch = OrderedDict([
			("linear1", c.Linear(3, 10)),
			("relu1", nn.Tanh()),
			("linear2", c.Linear(10, 10)),
			("relu2", nn.Tanh
			()),
			("linear3", c.Linear(10, 1)),
		])
		return nn.Sequential(self.arch), 140

	def reset(self):
		self.network, _ = self._build_network()

	def update_weights(self, idx, updates):
		prev_weight_idx = 0
		for name, weight in self.arch.items():
			try:
				cur_no_weights = weight.in_features * weight.out_features
				weight.weight = 1 / (1 + torch.exp(-10 * (
						weight.weight + updates[idx, prev_weight_idx:prev_weight_idx + cur_no_weights, :].reshape(
					weight.in_features, weight.out_features) - .5)))
				prev_weight_idx += cur_no_weights
			except AttributeError:
				pass

	def forward(self, x, update):
		preds = torch.zeros([x.shape[0], 1])
		for idx, x_i in enumerate(x[:, 0, :]):
			self.update_weights(idx, update)
			preds[idx, :] = self.network(x_i)
		return preds


class Updater(nn.Module):
	def __init__(self, input_size, fast_net):
		super(Updater, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_size, 200),
			nn.Tanh(),
			nn.Linear(200, 200),
			nn.Tanh(),
			nn.Linear(200, fast_net.no_fast_weights)
		)
		self.relu = nn.ReLU()
		self.fast_net = fast_net

	def forward(self, x):
		network_updates = torch.zeros([x.shape[0], self.fast_net.no_fast_weights, 1])
		for idx, x_i in enumerate(x[:, 0, :]):
			network_updates[idx, :] = self.network(x_i).unsqueeze(1)

		pred = self.fast_net(x, network_updates)
		return pred


class RNNBaseline(nn.Module):
	def __init__(self):
		super(RNNBaseline, self).__init__()
		self.rnn = nn.RNN(3, 10)
		self.out_weights = nn.Linear(10, 1)

	def forward(self, x):
		out, h = self.rnn(x)
		out_preds = torch.zeros([x.shape[0], 1])
		for idx, h in enumerate(out):
			out_preds[idx, :] = self.out_weights(h)
		return out_preds


