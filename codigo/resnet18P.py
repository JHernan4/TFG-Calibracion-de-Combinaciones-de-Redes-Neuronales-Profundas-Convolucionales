import torch
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
import torch.nn as nn

import numpy as np
import os


def lr_scheduler(epoch):
	if epoch < 150:
		return 0.1
	elif epoch < 250:
		return 0.01
	elif epoch < 350:
		return 0.001

if __name__ == '__main__':

	#establecemos semilla inicial para la generacion de las semillas torch
	np.random.seed(123)

	#creacion de las transformaciones que aplicaremos sobre el dataset cifar10
	print('==> Preparing data...')
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

	#creamos los dataloaders para iterar el conjunto de datos
	train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=128,shuffle=True,num_workers=workers)
	test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=128,shuffle=False,num_workers=workers)

    print('==> Building model..')
    resnet18 = models.resnet18(False)
	loss = nn.CrossEntropyLoss()
	scheduler=lr_scheduler
	for e in range(350):
		ce_test,MC,ce=[0.0]*3
		optimizer=torch.optim.adam(resnet18.parameters(),lr=scheduler(e),momentum=0.9)
		for x,t in train_loader:
			x,t=x.cuda(),t.cuda()
			resnet18.train()
			o=resnet18.forward(x)
			cost=loss(o,t)
			cost.backward()
			optimizer.step()
			optimizer.zero_grad()
			ce+=cost.data

		with torch.no_grad():
			for x,t in test_loader:
				x,t=x.cuda(),t.cuda()
				resnet18.test()
				test_pred=resnet18.forward(x)
				index=torch.argmax(test_pred,1) #compute maximum
				MC+=(index!=t).sum().float() #accumulate MC error

		print("Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/500.,100*MC/10000.))
