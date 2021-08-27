import numpy.random
import torch
import numpy as np
import math

options = [
	torch.tensor([1, 0, 0]),
	torch.tensor([0, 1, 0]),
	torch.tensor([0, 0, 1])
]


def load_episode(episode_len):
	# np.random.seed(seed)
	percent = np.random.random(1)
	indicies = np.random.randint(1, episode_len, math.floor(episode_len * percent))
	indicies.sort()
	#
	to_del = []
	# No contiguous values
	for idx in range(len(indicies) - 2):
		if abs(indicies[idx] - indicies[idx + 1]) == 1:
			to_del.append(idx)

	indicies = np.delete(indicies, to_del)

	data = np.random.randint(0, 3, episode_len)
	x = torch.zeros([episode_len, 3]).long()

	pos_ind = []
	a_tracker = 0
	for idx, _ in enumerate(x):
		if idx in indicies:
			a_tracker = 1
			x[idx - 1, :] = options[0]
			x[idx, :] = options[1]
			if idx-1 in pos_ind:
				pos_ind.pop()
		else:
			x[idx, :] = options[data[idx]]
		if x[idx, 0] == 1:
			a_tracker = 1
		elif x[idx, 1] == 1 and a_tracker == 1:
			a_tracker = 0
			pos_ind.append(idx)
		else:
			a_tracker = 0

	# for idx, _ in enumerate(x):
	# 	x[idx, :] = options[data[idx]]
	# 	if x[idx, 0] == 1:
	# 		pos_ind.append(idx)

	y = torch.zeros([episode_len])
	y[pos_ind] = 1
	return x.float().cpu(), y.float().cpu()

