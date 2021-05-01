import torch
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
from torch import nn

import numpy
import os

class ResNet18():

	def __init__(self, pretrained=False):
		self = models.resnet18(pretrained = pretrained)
		self.CE = nn.CrossEntropyLoss()

		def forward_train(self, x):
			self.train()
			return self.forward(x)

		def forward_test(self, x):
			self.test()
			return self.forward(x)

		def Loss(self, t_, t):
			return self.CE(t_, t)

def lr_scheduler(epoch):
	if epoch < 150:
		return 0.1
	elif epoch < 250:
		return 0.01
	elif epoch < 350:
		return 0.001

if __name__ == '__main__':

	#Definimos las transformaciones que van a aplicarse sobre el dataset
	cifar10_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
	                   transforms.RandomHorizontalFlip(),
	                   transforms.ToTensor(),
	                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transforms are different for train and test

	cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
	                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


	#Creamos el dataset (en este caso importamos el CIFAR10) y aplicamos las transformaciones definidas previamente
	workers = (int)(os.popen('nproc').read())
	cifar10_train=datasets.CIFAR10('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
	cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)

	#por ultimo cargamos los datasets y la configuracion con que van a ser usados
	train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=100,shuffle=True,num_workers=workers)
	test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)

	#definimos el modelo de red neuronal y lo movemos a la GPU
	resnet18 = ResNet18(True)
	resnet18.cuda()

	scheduler=lr_scheduler
	#iteramos cada una de las epocas

	for e in range(5):
		ce_test,MC,ce=[0.0]*3 #se definen las variables
		#creamos el optimizador
		optimizer=torch.optim.SGD(resnet18.parameters(),lr=scheduler(e),momentum=0.9)

		for x,t in train_loader: #sample one batch
			x,t=x.cuda(),t.cuda()
			o=myNet.forward_train(x)
			cost=myNet.Loss(o,t)
			cost.backward()
			optimizer.step()
			optimizer.zero_grad()
			ce+=cost.data

		with torch.no_grad():
			for x,t in test_loader:
				x,t=x.cuda(),t.cuda()
				test_pred=myNet.forward_test(x)
				index=torch.argmax(test_pred,1) #compute maximum
				MC+=(index!=t).sum().float() #accumulate MC error

		print("Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/500.,100*MC/10000.))		
