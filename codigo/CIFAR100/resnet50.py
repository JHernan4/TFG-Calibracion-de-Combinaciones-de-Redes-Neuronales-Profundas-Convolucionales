import torch
if not torch.cuda.is_available():
	print("Error al cargar GPU")
	exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
from torch import nn
import sys
sys.path.append("../models")
from resnet import ResNet50

import numpy as np
import os
import sys
import random

file = "CIFAR100_resnet18_seed_"

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
	if len(sys.argv) < 5:
		print("Numero de parámetros incorrecto")
		exit(-1)
	if sys.argv[1] != "--seed":
		print("Parametro {} incorrecto".format(sys.argv[1]))
		exit(-1)
	if sys.argv[3] != "--nEpocas":
		print("Parametro {} incorrecto".format(sys.argv[3]))
	
	seed = int(sys.argv[2])
	fileName = file+sys.argv[2]+".dat"
	nEpocas = int(sys.argv[4])
	f=open(fileName, "w")
	f.write("Epoca\tCrossEntropy\tAccuracy\n")
	print("Fichero {} creado para salida de datos".format(fileName))
	scheduler = lr_scheduler
	print("==> Preparing data...")
	cifar100_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
					   	transforms.ToTensor(),
					   	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 

	cifar100_transforms_test=transforms.Compose([transforms.ToTensor(),
					   	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


	workers = (int)(os.popen('nproc').read())
	cifar100_train=datasets.CIFAR100('/tmp/',train=True,download=True,transform=cifar100_transforms_train)
	cifar100_test=datasets.CIFAR100('/tmp/',train=False,download=False,transform=cifar100_transforms_test)
	loss = nn.CrossEntropyLoss()
	
	train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=100,shuffle=True,num_workers=workers, worker_init_fn=seed_worker)
	test_loader = torch.utils.data.DataLoader(cifar100_test,batch_size=100,shuffle=False,num_workers=workers, worker_init_fn=seed_worker)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	print("==> Building model")
	resnet50 = ResNet50(100)
	resnet50.cuda()
	for e in range(nEpocas):
		ce=0.0
		optimizer=torch.optim.SGD(resnet50.parameters(),lr=scheduler(e),momentum=0.9)
		resnet50.train()
		for x,t in train_loader:
			x,t=x.cuda(),t.cuda()
			o=resnet50.forward(x)
			cost=loss(o,t)
			cost.backward()
			optimizer.step()
			optimizer.zero_grad()
			ce+=cost.data

		with torch.no_grad():
			correct,total=0,0
			resnet50.eval()
			for x,t in test_loader:
				x,t=x.cuda(),t.cuda()
				test_pred=resnet50.forward(x)
				index=torch.argmax(test_pred,1)
				total+=t.size(0)
				correct+=(index==t).sum().float()

		print("Epoca {}: cross entropy {:.5f} and accuracy {:.3f}".format(e,ce/500.,100*correct/total))
		f.write(str(e)+"\t"+str(round(ce.item()/500.,4))+"\t\t"+str(round(100*correct.item()/total,2))+"\n")

	f.close()
	print("***Fin de ejecución")