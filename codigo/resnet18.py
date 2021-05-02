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


def lr_scheduler(epoch):
	if epoch < 150:
		return 0.1
	elif epoch < 250:
		return 0.01
	elif epoch < 350:
		return 0.001

if __name__ == '__main__':

	#definimos el modelo de red neuronal y lo movemos a la GPU
	resnet18 = models.resnet18(True)
	resnet18.cuda()

	cifar10_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
	                   transforms.RandomHorizontalFlip(),
	                   transforms.ToTensor(),
	                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #transforms are different for train and test

	cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
	                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


	#Second, you create your dataset Dataset.  Cifar10 is also provided so we just use it. If you create your own dataset  you can decide how it is loaded to memory and which transformations do you want to apply. Check my tutorial on transfer learning. Basically you use a similar tool to torch.nn but designed for datasets.

	workers = (int)(os.popen('nproc').read())
	cifar10_train=datasets.CIFAR10('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
	cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)

	#Third your dataloader. You just pass any dataset you have created. For instance you can decide to shuffle all the dataset at each iteration (that improves generalization) and also yo use several threads. In this case I will detect how many threads does my machine have and use them. Each thread loads a batch of data in parallel to your main loop (your CNN training)
	train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=100,shuffle=True,num_workers=workers)
	test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)

	loss = nn.CrossEntropyLoss()
	seeds = []
	losses = []
	testErrors = []
	#generamos 5 semillas aleatorias
	for i in range(5):
		seeds.append(np.random.randint(150))
	#para cada semilla realizamos el entrenamiento y clasificacion del modelo
	for seed in seeds:
		torch.cuda.manual_seed(123)
		scheduler=lr_scheduler
		for e in range(350):
			ce_test,MC,ce=[0.0]*3
			optimizer=torch.optim.SGD(resnet18.parameters(),lr=scheduler(e),momentum=0.9)
			for x,t in train_loader: #sample one batch
				x,t=x.cuda(),t.cuda()
				o=resnet18.forward(x)
				cost=loss(o,t)
				cost.backward()
				optimizer.step()
				optimizer.zero_grad()
				ce+=cost.data

				''' You must comment from here'''
				with torch.no_grad():
					for x,t in test_loader:
						x,t=x.cuda(),t.cuda()
						test_pred=resnet18.forward(x)
						index=torch.argmax(test_pred,1) #compute maximum
						MC+=(index!=t).sum().float() #accumulate MC error

			print("Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/500.,100*MC/10000.))

		losses.append(ce/500.)
		testErrors.append(100*MC/10000.)
		print("--------------------------------------------------------")
		print("--------------------------------------------------------")
	print(">>>> Resultados: ")
	for i in range(len(seeds)):
		print("\tModelo {}: cross entropy {:.5f} and Test error {:.3f}".format(i+1, losses[i], testErrors[i]))
