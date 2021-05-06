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
    #creacion de las transformaciones que aplicaremos sobre el dataset cifar10
	cifar10_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
	                   transforms.RandomHorizontalFlip(),
	                   transforms.ToTensor(),
	                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transforms are different for train and test

	cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
	                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


	#cargamos el dataset CIFAR10
	workers = (int)(os.popen('nproc').read())
	cifar10_train=datasets.CIFAR10('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
	cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)

    train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=100,shuffle=True,num_workers=workers)
    test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)

    model = ResNet18()
    net = torch.nn.DataParallel(model, device_ids=[0,1])
    for x,t in train_loader:
        x,t=x.cuda(),t.cuda()
        output = net(x)
        print(output)
