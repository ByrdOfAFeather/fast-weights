import torch
import torch.nn as nn
import numpy as np
import components as c
import torch.nn.functional as F

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
		]

		def forward(x):
			cur = x
			for module in self.prototype:
				cur = module[1](cur)
			return cur

		return forward, self._count_weights(self.prototype)


class LinearModel(nn.Module):
	def __init__(self, input_dim: int, output_dim: int):
		super().__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.linear(x)


class LinearWithAdapter(nn.Module):
	def __init__(self, underlying_model, input_dim, output_dim):
		super().__init__()
		self.underlying_model = underlying_model
		self.adapter = c.Linear(input_dim, output_dim)
		self.no_fast_weights = self.adapter.no_fast_weights

	@staticmethod
	def update_func(weight, update):
		return 1 / (1 + torch.exp(-10 * (weight + update - .5)))

	def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
		self.underlying_model.load_state_dict(state_dict, strict)

	def update_weights(self, weight_update):
		self.adapter.update_weights(weight_update, 0, self.update_func)

	def reset_weights(self):
		self.adapter.reset()

	def forward(self, x):
		# underlying_model_out = self.underlying_model(x)
		# underlying_model_out = torch.rand(self.adapter.weight.shape[0])
		underlying_model_out = torch.tensor([-0.5, 1, .5])
		# underlying_model_out = x[:, 0:3]
		return F.linear(underlying_model_out, self.adapter.weight)


class Updater(nn.Module):
	def __init__(self, model, input_size, output_size=None):
		super().__init__()
		self.model = model
		if output_size:
			self.linear_1 = nn.Linear(input_size, output_size)
		else:
			# self.linear_1 = nn.Linear(input_size, int(128 / 2))
			self.linear_1 = nn.Linear(input_size, self.model.no_fast_weights)
		self.relu = nn.ReLU()
		self.linear_2 = nn.Linear(9, int(128 / 2))
		self.linear_3 = nn.Linear(128, self.model.no_fast_weights)

	def reset_weights(self):
		self.model.reset_weights()

	def forward(self, updater_info, x):
		updates = self.linear_1(updater_info)
		# weight_res = self.linear_2(self.model.adapter.weight.detach().flatten().unsqueeze(0))
		# updates = self.linear_3(torch.concat([input_res, weight_res], dim=1))
		self.model.update_weights(updates)
		return self.model(x)
