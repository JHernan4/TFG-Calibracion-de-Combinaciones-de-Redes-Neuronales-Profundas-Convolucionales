###########################################################################################################
#                                       ensembleResnet50.py
#programa que carga tantos modelos como se indique con el parametro nModelos (de su correspondiente .pt)
#realiza su explotación para el dataset CIFAR100 y calcula el accuracy individual de cada uno de ellos. 
# Finalmente, calcula el accuracy del average de todos estos modelos combinados (ensemble)
###########################################################################################################

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
from densenet import DenseNet121
import numpy as np
from numpy import array
import os
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parametros para configuracion del entrenamiento de las redes neuronales convolucionales')
    parser.add_argument('--nModelos', help="número de modelos que componen el emsemble", required=True, type = int)
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
    worker_seed=torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def explotation(model, testLoader, n, path):
    softmax = nn.Softmax(dim=1)
    path =  path + "_"+str(n+1) + '.pt'
    logits = [] #para guardar los logits del modelo 
    logitsSof = [] #almacena los logits pasados por la Softmax para devolverlos y usarlos en el average
    with torch.no_grad():
        correct,total=0,0
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            test_pred=model.forward(x)
            logits.append(test_pred)
            logit=softmax(test_pred).cpu()
            logitsSof.append(logit)
            index=torch.argmax(logit,1)
            total+=t.size(0)
            correct+=(t==index.cuda()).sum().float()
    
    print("Modelo {}: accuracy {:.3f}".format(n+1, 100*(correct/total)))
    torch.save(logits, path)
    print("Logits del modelo {} guardados correctamente en el fichero {}".format(n+1, path))
    return logitsSof


def avgEnsemble(logits, testLoader):
    avgLogits = []
    for i in range(len(logits[0])):
        avgLogits.append(logits[0][i]/len(logits))
    
    for n in range(1, len(logits)):
        for i in range(len(logits[n])):
            avgLogits[i]+=logits[n][i]/len(logits)
    
    with torch.no_grad():
        correct,total=0,0
        i=0
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            total+=t.size(0)
            index=torch.argmax(avgLogits[i],1)
            correct+=(t==index.cuda()).sum().float()
            i=i+1

    return correct/total

if __name__ == '__main__':
    args = parse_args()
    PATH = './checkpoint'+'_densenet121'
    LOGITSPATH = './logits_densenet121'
    nModelos = args.nModelos

    workers = (int)(os.popen('nproc').read())
    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)
    test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)

    logits = []
    for n in range(nModelos):
        model = DenseNet121()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))
        model.eval()
        logits.append(explotation(model, test_loader, n, LOGITSPATH))

    avgACC = avgEnsemble(logits, test_loader)

    print("Ensemble de {} modelos: {:.3f}".format(nModelos, 100*avgACC))
