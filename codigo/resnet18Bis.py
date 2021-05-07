import torch
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
from torch import nn
from models.resnetBis import resnet18, resnet50

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
    cifar100_transforms_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar100_transforms_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    workers = (int)(os.popen('nproc').read())
    cifar100_train = datasets.CIFAR100('/tmp', train=True, download=True, transform=cifar100_transforms_train)
    cifar100_test = datasets.CIFAR100('/tmp', train=False, download=True, transform=cifar100_transforms_test)

    loss = nn.CrossEntropyLoss()
    seed = np.random.randint(2**10)
    resnet18 = resnet18()
    net = torch.nn.DataParallel(resnet18, device_ids=[0,1]).cuda()
    for n in range(nModelos):
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
                    x,t=x.cuda(), t.cuda()
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

    print(">>>Averages: cross entropy {:.5f} and accuracy {:.3f}".format(avgCE, avgACC))
