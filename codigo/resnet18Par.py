import torch
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
from torch import nn
from models.resnet import ResNet18

import numpy as np
import os
import random


def lr_scheduler(epoch):
	if epoch < 150:
		return 0.1
	elif epoch < 250:
		return 0.01
	elif epoch < 350:
		return 0.001

def seed_worker(worker_id):
	worker_seed=torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

if __name__ == '__main__':
    nEpocas = 200
    nModelos = 5
    scheduler = lr_scheduler
    print("==> Preparing data...")

    model = ResNet18()
    net = torch.nn.DataParalell(model, device_ids=[0,1])
    output = net(0.65)
    print(output)
