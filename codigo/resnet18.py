from resnet18 import *
import torch
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)
import torchvision #computer vision dataset module
from torchvision import datasets,transforms

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
    #1. Definimos las transformaciones del dataset (CIFAR10)
    
    cifar10_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    #2. Creamos dataset de entrenamiento y de validacion
    workers = (int)(os.popen('nproc').read()) 
    cifar10_train=datasets.CIFAR10('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)

    #3. Cargamos los datasets
    train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=100,shuffle=True,num_workers=workers)
    test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)

    #4. Creamos el modelo (resnet18)
    nn = resnet18()

    scheduler=lr_scheduler
    for e in range(5):
	    ce_test,MC,ce=[0.0]*3
	    #now create and optimizer
	    optimizer=torch.optim.SGD(myNet.parameters(),lr=scheduler(e),momentum=0.9)

	    for x,t in train_loader: #sample one batch
		    x,t=x.cuda(),t.cuda()
		    o=nn.forward_train(x) 
		    cost=nn.Loss(o,t) 
		    cost.backward() 
		    optimizer.step()
		    optimizer.zero_grad()
		    ce+=cost.data

	    ''' You must comment from here'''
	    with torch.no_grad():
		    for x,t in test_loader:
			    x,t=x.cuda(),t.cuda()
			    test_pred=myNet.forward_test(x)
			    index=torch.argmax(test_pred,1) #compute maximum
			    MC+=(index!=t).sum().float() #accumulate MC error
	
	    print("Epoch {} cross entropy {:.5f} and Test error {:.3f}".format(e,ce/500.,100*MC/10000.))