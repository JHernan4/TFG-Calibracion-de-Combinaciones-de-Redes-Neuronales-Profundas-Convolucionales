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
	accuracies = []
	crossEntropies = []
	scheduler = lr_scheduler
	print("==> Preparing data...")
	cifar100_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
					   	transforms.ToTensor(),
					   	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transforms are different for train and test

	cifar100_transforms_test=transforms.Compose([transforms.ToTensor(),
					   	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


	#Second, you create your dataset Dataset.  Cifar10 is also provided so we just use it. If you create your own dataset  you can decide how it is loaded to memory and which transformations do you want to apply. Check my tutorial on transfer learning. Basically you use a similar tool to torch.nn but designed for datasets.
	workers = (int)(os.popen('nproc').read())
	cifar100_train=datasets.CIFAR100('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
	cifar100_test=datasets.CIFAR100('/tmp/',train=False,download=False,transform=cifar10_transforms_test)
	loss = nn.CrossEntropyLoss()
	for n in range(nModelos):
		#Third your dataloader. You just pass any dataset you have created. For instance you can decide to shuffle all the dataset at each iteration (that improves generalization) and also yo use several threads. In this case I will detect how many threads does my machine have and use them. Each thread loads a batch of data in parallel to your main loop (your CNN training)
		train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=100,shuffle=True,num_workers=workers, worker_init_fn=seed_worker)
		test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,shuffle=False,num_workers=workers, worker_init_fn=seed_worker)
		seed = np.random.randint(2**10)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		print("==> Building model {} (seed = {})...".format(n+1, seed))
		resnet18 = ResNet18()
		net = torch.nn.DataParallel(resnet18, device_ids=[0,1]).cuda()
		for e in range(nEpocas):
			ce=0.0
			optimizer=torch.optim.SGD(resnet18.parameters(),lr=scheduler(e),momentum=0.9)
			for x,t in train_loader:
				x,t=x.cuda(),t.cuda()
				o=net(x)
				cost=loss(o,t)
				cost.backward()
				optimizer.step()
				optimizer.zero_grad()
				ce+=cost.data

			with torch.no_grad():
				correct,total=0,0
				for x,t in test_loader:
					x,t=x.cuda(),t.cuda()
					test_pred=net(x)
					index=torch.argmax(test_pred,1)
					total+=t.size(0)
					correct+=(index==t).sum().float()

			print("Epoca {}: cross entropy {:.5f} and accuracy {:.3f}".format(e,ce/500.,100*correct/total))
		accuracies.append(100*correct/total)
		crossEntropies.append(ce/500.)
		print("-----------------------------------------------------------------")
		print("-----------------------------------------------------------------")

	print(">>> Resultados: ")
	avgCE=0.0
	avgACC=0.0
	for i in range(len(accuracies)):
		print("\tModelo {}: cross entropy {:.5f} and accuracy {:.3f}".format(i+1,crossEntropies[i],accuracies[i]))
		avgCE+=crossEntropies[i]/len(crossEntropies)
		avgACC+=accuracies[i]/len(accuracies)

	print(">>>Averages: cross entropy {:.5f} and accuracy {:.3f}".format(avgCE, avgACC)
