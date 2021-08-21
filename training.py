from toy_example.data import load_episode


def train(model, criterion, optim, episode_len):
	for i in range(1000):
		x, y = load_episode(episode_len)
		optim.zero_grad()
		preds = model(x.float().unsqueeze(1))
		cur_loss = criterion(preds.view(-1), y)
		cur_loss.backward()
		optim.step()
		print(cur_loss)
		model.fast_net.reset()