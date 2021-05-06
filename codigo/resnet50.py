import torch
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
from torch import nn
from models.resnet import ResNet18, ResNet50

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
	worker_seed=torch.initial.seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

if __name__ == '__main__':
	nEpocas = 250
	nModelos = 3
	scheduler=lr_scheduler
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

	loss = nn.CrossEntropyLoss()
	print("==> Building model...")
	seeds = []
	accuracies = []
	crossEntropies = []
	for i in range(nModelos):
		#creamos los dataloaders para iterar el conjunto de datos
		train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=100,shuffle=True,num_workers=workers, worker_init_fn=seed_worker)
		test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers, worker_init_fn=seed_worker)
		seed = np.random.randint(2**32)
		seeds.append(seed)
		print("Semilla: {}".format(seed))
		torch.manual_seed(seed)
		resnet50 = ResNet50()
		resnet50.cuda()
		for e in range(nEpocas):
			ce = 0.0
			optimizer=torch.optim.SGD(resnet50.parameters(),lr=scheduler(e),momentum=0.9)
			for x,t in train_loader:
				x,t=x.cuda(),t.cuda()
				resnet50.train()
				o=resnet50.forward(x)
				cost=loss(o,t)
				cost.backward()
				optimizer.step()
				optimizer.zero_grad()
				ce+=cost.data

			with torch.no_grad():
				correct = 0
				total = 0
				for x,t in test_loader:
					x,t=x.cuda(),t.cuda()
					resnet50.eval()
					test_pred=resnet50.forward(x)
					index=torch.argmax(test_pred,1)
					total+=t.size(0)
					correct+=(index==t).sum().float()

			print("Epoca {}: cross entropy {:.5f} and accuracy {:.3f}".format(e,ce/500.,100*correct/total))

		crossEntropies.append(ce/500)
		accuracies.append(100*correct/total)
		print("---------------------------------------------------")
		print("---------------------------------------------------")

	avgCE = 0.0
	avgACC = 0.0
	print(">>>Resultados: ")
	for i in range(len(seeds)):
		print("\tModelo {} (semilla {}): cross entropy {:.5f} and accuracy {:.3f}".format(i+1, seeds[i], crossEntropies[i], accuracies[i]))
		avgCE+=crossEntropies[i]/len(crossEntropies)
		avgACC+=accuracies[i]/len(accuracies)

	print(">>>Valores medios finales: cross entropy {:.5f} and accuracy {:.3f}".format(avgCE, avgACC))
