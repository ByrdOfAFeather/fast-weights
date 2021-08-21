import torch
import torch.nn as nn
import numpy as np

np.random.seed(225531)
torch.manual_seed(225531)

from toy_example.data import load_episode


class FastNetwork(nn.Module):
	def __init__(self):
		super(FastNetwork, self).__init__()
		self.network = (-1 - 1) * torch.rand([3, 1]) + 1
		self.temp_update = None
		self.sigmoid = nn.Sigmoid()

	def reset(self):
		self.network = (-1 - 1) * torch.rand([3, 1]) + 1

	def forward(self, x, update):
		preds = torch.zeros([x.shape[0], 1])
		for idx, x_i in enumerate(x[:, 0, :]):
			self.network = 1 / (1 + torch.exp(-10 * (self.network + update[idx, :, :] - .5)))
			preds[idx, :] = torch.matmul(x_i, self.network)
		return preds


class Updater(nn.Module):
	def __init__(self):
		super(Updater, self).__init__()
		self.network = nn.Linear(3, 3)
		self.relu = nn.ReLU()
		self.fast_net = FastNetwork()

	def forward(self, x):
		network_updates = torch.zeros([x.shape[0], 3, 1])
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


def train():
	loss = nn.MSELoss(reduction="mean")
	EPISODE_LEN = 300
	updater = Updater()
	optim = torch.optim.Adam([i for name, i in updater.named_parameters() if "fast_net" not in name], lr=.1)
	for i in range(1000):
		x, y = load_episode(EPISODE_LEN)
		optim.zero_grad()
		preds = updater(x.float().unsqueeze(1))
		cur_loss = loss(preds.view(-1), y)
		cur_loss.backward()
		optim.step()
		print(cur_loss)
		updater.fast_net.reset()


train()
