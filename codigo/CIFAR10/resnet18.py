##################################################################################################################
#programa 1 para la creacion de ensembles de resnet18. Crea y entrena para el dataset CIFAR10 tantos modelos
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
from resnet import ResNet18
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse

def lr_scheduler(epoch):
    if epoch < 150:
        return 0.1
    elif epoch < 250:
        return 0.01
    elif epoch < 350:
        return 0.001

scheduler = lr_scheduler

class MyModel():

    def __init__(self, model, nEpocas=250):
        self.net = model
        self.trainAccuracies = np.empty(nEpocas)
        self.validationAccuracies = np.empty(nEpocas)

    def trainModel(self, trainLoader, validationLoader, seed, nModelo, path, nEpocas=250):
        loss = nn.CrossEntropyLoss()
        sm = nn.Softmax(dim=1)
        torch.manual_seed(seed)
        
        path = path + "_"+str(nModelo+1) + '.pt'
        for e in range(nEpocas):
            print("Epoca {}/{}".format(e+1, nEpocas))
            correctT, totalT, correctV, totalV = 0,0,0,0
            ceT, ceV = 0,0
            optimizer=torch.optim.SGD(model.parameters(),lr=scheduler(e),momentum=0.9)
            self.net.train()
            counter=0
            for x,t in trainLoader:
                x,t=x.cuda(),t.cuda()
                pred=self.net.forward(x)
                cost=loss(pred,t)
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
                ceT+=cost.data
                counter+=1

                with torch.no_grad():
                    pred = sm(pred)
                    index = torch.argmax(pred, 1)
                    totalT+=t.size(0)
                    correctT+=(t==index).sum().float()
            
            print("\tTrain accuracy: {}".format(ceT/counter))
            self.trainAccuracies[e] = correctT/totalT
            counter=0
            for x,t in validationLoader:
                with torch.no_grad():
                    x,t=x.cuda(),t.cuda()
                    pred=self.net.forward(x)
                    costV = loss(pred,t)
                    pred = sm(pred)
                    index = torch.argmax(pred, 1)
                    totalV+=t.size(0)
                    correctV+=(t==index).sum().float()
                    ceV+=costV.data
                    counter+=1
            
            print("\tValidation loss: {}".format(ceV/counter))
            self.validationAccuracies[e] = correctV/totalV
        
        torch.save(self.net.state_dict(), path)
        print("Modelo {} guardado correctamente en {}".format(nModelo+1, path))	
        return self.net

    def saveGraficas(self, nModelo, nEpocas=250):
        file = 'checkpointResnet18Tra/resnet18_'+str(nModelo+1)+'.jpg'
        x = np.linspace(0,nEpocas,nEpocas)
        plt.figure()
        plt.plot(x, self.trainAccuracies, label="Train")
        plt.plot(x, self.validationAccuracies, label="Validation")
        
        plt.xlabel("Number of epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Resnet18 on CIFAR10")
        plt.legend()
        plt.savefig(file)
 
def parse_args():
    parser = argparse.ArgumentParser(description='Parametros para configuracion del entrenamiento de las redes neuronales convolucionales')
    parser.add_argument('--seed', help='semilla para inicializar generador de numeros aleatorios de numpy', required=True, type=int)
    parser.add_argument('--nEpocas', help='número de epocas para el entrenamiento de las redes neuronales', required=True, type=int)
    parser.add_argument('--nModelos', help="número de modelos que componen el emsemble", required=True, type = int)
    args = parser.parse_args()
    return args

def separarConjunto(dataset, trainSize=10000):
    val_set, train_set = torch.utils.data.random_split(dataset, [len(dataset)-trainSize, trainSize])
    print(len(val_set))
    print(len(train_set))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=False, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=workers)

    return train_loader, val_loader


def seed_worker(worker_id):
    worker_seed=torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



if __name__ == '__main__':
    PATH = './checkpointResnet18Tra/checkpoint'+'_resnet18'
    args = parse_args()
    nModelos = args.nModelos
    nEpocas = args.nEpocas
    
    
    print("==> Preparando dataset de entrenamiento...")
    cifar10_transforms_train=transforms.Compose([transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    workers = (int)(os.popen('nproc').read())
    cifar10_train=datasets.CIFAR10('/tmp/',train=True,download=True,transform=cifar10_transforms_train)
    
       
    train_loader, validation_loader = separarConjunto(cifar10_train)
    
    print("==> Entrenando modelos...")
    models=[]
    for n in range(nModelos):
        model=ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        net = MyModel(model, nEpocas)
        seed = np.random.randint(2**10)
        models.append(net.trainModel(train_loader, validation_loader, seed, n, PATH, nEpocas))

        net.saveGraficas(n, nEpocas)
