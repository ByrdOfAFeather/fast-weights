from datetime import datetime
import torch.optim
import torch.nn as nn
from regression_models import FeedForwardFastNet, BruteForceUpdater, RNNBaseline, FromToUpdater, RNNFastNet, RNNUpdater, FeedForwardBaseline
from data import load_episode
torch.autograd.set_detect_anomaly(True)
EPISODE_LEN = 100


def train(model, criterion, optim, episode_len, device="cpu"):
	avg_time = 0
	for i in range(100):
		start = datetime.now()
		x, y = load_episode(episode_len)
		if device == "cuda":
			x = x.cuda()
			y = y .cuda()
		optim.zero_grad()
		preds = model(x.float().unsqueeze(1))
		cur_loss = criterion(preds.view(-1), y)
		cur_loss.backward()
		# print(((preds.view(-1) - y) ** 2 > .05).any())
		# print([(name, i.grad) for name, i in model.named_parameters()])
		# print(preds)
		# print(y)
		optim.step()
		try:
			# Sometimes we are using this function to train a non-fast weight model
			model.fast_net.reset()
		except AttributeError:
			pass
		end = datetime.now()
		avg_time += (end - start).total_seconds()

	avg_loss = 0
	for i in range(100):
		model.eval()
		x, y = load_episode(episode_len)
		if device == "cuda":
			x = x.cuda()
			y = y.cuda()
		preds = model(x.float().unsqueeze(1))
		cur_loss = criterion(preds.view(-1), y)
		avg_loss += cur_loss
		try:
			# Sometimes we are using this function to train a non-fast weight model
			model.fast_net.reset()
		except AttributeError:
			pass

	print(F"Average loss for model during eval: {avg_loss / 100}")


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


def feed_forward_base_line():
	baseline = FeedForwardBaseline(device="cuda")
	criterion = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(baseline.parameters(), lr=.001)
	train(baseline, criterion, optimizer, 300, device="cuda")


def feed_forward_fast_weights():
	fast_net = FeedForwardFastNet(device="cuda")
	updater = BruteForceUpdater(fast_net=fast_net, input_size=3, device="cuda")
	criterion = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(updater.parameters(), lr=.001)
	train(updater, criterion, optimizer, 300, device="cuda")

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
	feed_forward_base_line()
	feed_forward_fast_weights()
	# rnn_exper()
	# print("==================")
	# rnn_baseline()
	# print("===============")
	# feed_forward_fast_weights()
	# print("=============")