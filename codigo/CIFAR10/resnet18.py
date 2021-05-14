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
from resnet import ResNet18
import numpy as np
from numpy import array
import os
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Parametros para configuracion del entrenamiento de las redes neuronales convolucionales')
    parser.add_argument('--seed', help='semilla para inicializar generador de numeros aleatorios de numpy', required=True, type=int)
    parser.add_argument('--nEpocas', help='número de epocas para el entrenamiento de las redes neuronales', required=True, type=int)
    parser.add_argument('--nModelos', help="número de modelos que componen el emsemble", required=True, type = int)
    args = parser.parse_args()
    return args

PATH = './checkpoint'+'_resnet18'

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


def trainModel(trainLoader, seed, nModelo, nEpocas=250):
    torch.manual_seed(seed)
    model=ResNet18()
    PATH = PATH + str(nModelo) + '.pt'
    for e in range(nEpocas):
        ce=0.0
        optimizer=torch.optim.SGD(model.parameters(),lr=scheduler(e),momentum=0.9)
        model.train()
        for x,t in trainLoader:
            x,t=x.cuda(),t.cuda()
            o=model.forward(x)
            cost=loss(o,t)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print("Epoca {}/{}".format(e+1, nEpocas))
    
    torch.save(model.state_dict(), PATH)	
    return model


if __name__ == '__main__':

    args = parse_args()
    nModelos = args.nModelos
    nEpocas = args.nEpocas
    
    scheduler = lr_scheduler
    print("==> Preparing data...")
    cifar10_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 

    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    workers = (int)(os.popen('nproc').read())
    cifar10_train=datasets.CIFAR10('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)
    loss = nn.CrossEntropyLoss()
    
    train_loader = torch.utils.data.DataLoader(cifar10_train,batch_size=100,shuffle=True,num_workers=workers, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers, worker_init_fn=seed_worker)
    
    models=[]
    for n in range(nModelos):
        seed = np.random.randInt(2**10)
        models.append(trainModel(train_loader, seed, n, nEpocas))