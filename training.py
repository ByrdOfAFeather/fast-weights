import torch.optim
import torch.nn as nn
from models import FeedForwardFastNet, BruteForceUpdater, RNNBaseline, FromToUpdater
from toy_example.data import load_episode
EPISODE_LEN = 300


def train(model, criterion, optim, episode_len):
	for i in range(500):
		x, y = load_episode(episode_len)
		optim.zero_grad()
		preds = model(x.float().unsqueeze(1))
		cur_loss = criterion(preds.view(-1), y)
		cur_loss.backward()
		optim.step()
		print(cur_loss)
		try:
			# Sometimes we are using this function to train a non-fast weight model
			model.fast_net.reset()
		except AttributeError:
			pass


def feed_forward_fast_weights():
	fast_net = FeedForwardFastNet()
	updater = BruteForceUpdater(fast_net=fast_net, input_size=3)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(updater.parameters(), lr=.1)
	train(updater, criterion, optimizer, 300)


def feed_forward_fast_weights():
	fast_net = FeedForwardFastNet()
	updater = FromToUpdater(fast_net=fast_net, input_size=3)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(updater.parameters(), lr=.1)
	train(updater, criterion, optimizer, 300)


def rnn_baseline():
	model = RNNBaseline()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=.1)
	train(model, criterion, optimizer, 300)


if __name__ == "__main__":
	# feed_forward_fast_weights()
	# print("==================")
	# rnn_baseline()
	# print("===============")
	feed_forward_fast_weights()
	print("=============")