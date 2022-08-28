import torch
import torch.nn as nn
import numpy as np
import components as c
from collections import OrderedDict

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


class FeedForwardBaseline(nn.Module):
	def __init__(self, device):
		super(FeedForwardBaseline, self).__init__()
		self.device = device
		self.linear1 = nn.Linear(3, 100, device=self.device)
		self.linear2 = nn.Linear(100, 1, device=self.device)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		return self.sigmoid(self.linear2(self.linear1(x)))


class FeedForwardFastNet(FastNet):
	def __init__(self, device="cpu"):
		self.device = device
		super(FeedForwardFastNet, self).__init__()
		self.no_from = 6
		self.no_to = 4

	def _build_network(self):
		if self.prototype is not None:
			for name, mod in self.prototype:
				mod.reset()

		self.prototype = [
			("linear1", c.Linear(3, 100, device=self.device)),
			("linear2", c.Linear(100, 1, device=self.device)),
			("sigmoid", nn.Sigmoid())
		]

		def forward(x):
			cur = x
			for module in self.prototype:
				cur = module[1](cur)
			return cur

		return forward, self._count_weights(self.prototype)