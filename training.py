from datetime import datetime

import numpy as np
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

from models import LinearModel, Updater, LinearWithAdapter
from data import IrisDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
EPISODE_LEN = 100


def torch_input_dict(y):
	input_tensor = torch.zeros(y.shape[0], 3)
	for idx, y_cur in enumerate(y):
		if y_cur == 0:
			input_tensor[idx, :] = torch.tensor([1, 0, 0]).float()
		elif y_cur == 1:
			input_tensor[idx, :] = torch.tensor([0, 1, 0]).float()
		else:
			input_tensor[idx, :] = torch.tensor([0, 0, 1]).float()
	return input_tensor


def train_base_model(model, dataloader, dataloader_eval, criterion, optim, epochs=5, device="cpu", updater=False):
	idx = 0
	indx = []
	losses = []
	for i in range(epochs):
		with torch.no_grad():
			if updater: model.reset_weights()
			running_loss = 0
			accuracy_correct = 0
			accuracy_total = 0
			for x, y in dataloader_eval:
				if updater:
					# local_pred = model(torch.tensor(y).float().unsqueeze(1), x)
					local_pred = model(torch_input_dict(y), x)
				else:
					local_pred = model(x)
				running_loss += criterion(local_pred.view(-1, 3), y)
				print(np.argmax(local_pred))
				accuracy_correct += 1 if np.argmax(local_pred) == y else 0
				accuracy_total += 1
			print(f"Loss at {idx} : {running_loss}")
			print(f"CORRECT RESPONSES: {accuracy_correct / accuracy_total}")
			indx.append(idx)
			losses.append(running_loss)

		if updater: model.reset_weights()
		optim.zero_grad()
		for inputs, label in tqdm(dataloader):
			if updater:
				model_pred = model(torch_input_dict(label), x)
			else:
				model_pred = model(inputs)
			loss = criterion(model_pred.view(-1, 3), label)
			loss.backward(retain_graph=True)
			# torch.nn.utils.clip_grad_norm_([v for i, v in model.named_parameters() if "underlying_model" not in i],
			#                                1)
		optim.step()
		idx += 1
	plt.plot(indx, losses)
	plt.show()
	return model


if __name__ == "__main__":
	train, val, test = DataLoader(IrisDataset("IRIS_UPDATE.csv", .8),shuffle=True, batch_size=1), DataLoader(
		IrisDataset("IRIS_UPDATE.csv", .1), batch_size=1), DataLoader(IrisDataset("IRIS_UPDATE.csv", .1))
	model = LinearModel(4, 3)
	criterion = nn.CrossEntropyLoss()


	def base_model():
		optimizer = torch.optim.SGD(model.parameters(), lr=.00004)
		train_base_model(model, train, val, criterion, optimizer, epochs=100)
		torch.save(model.state_dict(), "base_model")


	def updater_model():
		model_adapter = LinearWithAdapter(model, 3, 3)
		model_adapter.load_state_dict(torch.load("base_model"))
		updater = Updater(model_adapter, 3, output_size=None)
		optimizer = torch.optim.SGD([v for i, v in updater.named_parameters() if "underlying_model" not in i], lr=.001)
		train_base_model(updater, train, val, criterion, optimizer, updater=True, epochs=1000)


	# base_model()
	updater_model()
