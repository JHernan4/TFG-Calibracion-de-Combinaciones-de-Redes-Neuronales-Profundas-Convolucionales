##################################################################################################################
#programa 1 para la creacion de ensembles de efficientnet. Crea y entrena para el dataset CIFAR100 tantos modelos
#como se indique por parametros (parametro --nModelos). Psteriormente vuelca cada modelo en su correspondiente .pt
#para ser posteriormente cargado por el programa ensembleResnet18.py 
##################################################################################################################

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
from efficientnet import EfficientNetB0
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


def trainModel(trainLoader, seed, nModelo, path, nEpocas=250):
    loss = nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    model=EfficientNetB0(100)
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    path = path + "_"+str(nModelo+1) + '.pt'
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
    
    torch.save(model.state_dict(), path)
    print("Modelo {} guardado correctamente en {}".format(nModelo+1, path))	
    return model


if __name__ == '__main__':
    PATH = './checkpointEfficientNetB0/checkpoint'+'_efficientnetB0'
    args = parse_args()
    nModelos = args.nModelos
    nEpocas = args.nEpocas
    
    scheduler = lr_scheduler
    print("==> Preparando dataset de entrenamiento...")
    cifar100_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    workers = (int)(os.popen('nproc').read())
    cifar100_train=datasets.CIFAR100('/tmp/',train=True,download=True,transform=cifar100_transforms_train)
       
    train_loader = torch.utils.data.DataLoader(cifar100_train,batch_size=100,shuffle=True,num_workers=workers, worker_init_fn=seed_worker)
    
    print("==> Entrenando modelos...")
    models=[]
    for n in range(nModelos):
        seed = np.random.randint(2**10)
        models.append(trainModel(train_loader, seed, n, PATH, nEpocas))