import torch.optim
import torch.nn as nn
from models import FeedForwardFastNet, BruteForceUpdater, RNNBaseline, FromToUpdater, RNNFastNet, RNNUpdater
from toy_example.data import load_episode
torch.autograd.set_detect_anomaly(True)
EPISODE_LEN = 100


def train(model, criterion, optim, episode_len, device="cpu"):
	for i in range(500):
		x, y = load_episode(episode_len)
		x.to(device)
		y.to(device)
		optim.zero_grad()
		preds = model(x.float().unsqueeze(1))
		cur_loss = criterion(preds.view(-1), y)
		cur_loss.backward()
		print(((preds.view(-1) - y) ** 2 > .05).any())
		# print([(name, i.grad) for name, i in model.named_parameters()])
		print(cur_loss)
		# print(preds)
		# print(y)
		optim.step()
		try:
			# Sometimes we are using this function to train a non-fast weight model
			model.fast_net.reset()
		except AttributeError:
			pass


def train_rnn(model, criterion, optim, episode_len, device="cpu"):
	for i in range(500):
		x, y = load_episode(episode_len)
		if device == "cuda":
			x, y = x.cuda(), y.cuda()
		optim.zero_grad()
		preds = model(x.float().unsqueeze(1))
		cur_loss = criterion(preds.view(-1), y)
		cur_loss.backward()
		# print([(name, i.grad) for name, i in model.named_parameters()])
		print(cur_loss)
		# print(preds)
		# print(y)
		optim.step()
		try:
			# Sometimes we are using this function to train a non-fast weight model
			model.fast_net.reset()
		except AttributeError:
			pass


def feed_forward_fast_weights():
	fast_net = FeedForwardFastNet()
	updater = BruteForceUpdater(fast_net=fast_net, input_size=3)
	criterion = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(updater.parameters(), lr=.01)
	train(updater, criterion, optimizer, 300)

def rnn_exper():
	fast_net = RNNFastNet(3, 2, device="cuda")
	updater = RNNUpdater(fast_net=fast_net, input_size=3, hidden_size=fast_net.no_fast_weights, device="cuda")
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(updater.parameters(), lr=.00001)
	train_rnn(updater, criterion, optimizer, EPISODE_LEN, device="cuda")

# def rnn_baseline():
# 	model = RNNBaseline()
# 	criterion = nn.MSELoss()
# 	optimizer = torch.optim.Adam(model.parameters(), lr=.1)
# 	train(model, criterion, optimizer, 300)


if __name__ == "__main__":
	# feed_forward_fast_weights()
	rnn_exper()
	# print("==================")
	# rnn_baseline()
	# print("===============")
	# feed_forward_fast_weights()
	# print("=============")