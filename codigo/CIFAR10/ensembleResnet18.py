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
sys.path.append("../calibration")
from resnet import ResNet18
from utils_calibration import compute_calibration_measures
import numpy as np
from numpy import array
import os
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parametros para configuracion del entrenamiento de las redes neuronales convolucionales')
    parser.add_argument('--nModelos', help="número de modelos que componen el emsemble", required=True, type = int)
    parser.add_argument('--L', help="constante para modificar logits", required=True, type = int)
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
    worker_seed=torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def explotation(model, testLoader, n, path):
    softmax = nn.Softmax(dim=1)
    logits = [] #para guardar los logits del modelo 
    logitsSof = [] #almacena los logits pasados por la Softmax para devolverlos y usarlos en el average
    path =  path + "_"+str(n+1) + '.pt'
    with torch.no_grad():
        correct,total=0,0
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            test_pred=model.forward(x)
            logit = softmax(test_pred)
            logitsSof.append(logit) #meter esto en la funcion de calibracion
            index = torch.argmax(logit, 1)
            logits.append(np.array(logit.cpu(), dtype=np.float32))
            total+=t.size(0)
            correct+=(t==index).sum().float()
    
    print("Modelo {}: accuracy {:.3f}".format(n+1, 100*(correct/total)))
    logits = np.array(logits)
    logitsSof = np.array(logitsSof)
    return logitsSof, logits


def calibracion(logits, n, targets):
    ECE,MCE,BRIER,NNL = 0.0,0.0,0.0,0.0

    ECE,MCE,BRIER,NNL = compute_calibration_measures(logits, targets, False, 100)

    print("Medidas de calibracion modelo {}: \n\tECE: {}\n\tMCE: {}\n\BRIER: {}\n\tNNL: {}".format(n+1, ECE, MCE, BRIER, NNL))


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
    PATH = './checkpointResnet18/checkpoint_resnet18'
    LOGITSPATH = './logitsResnet18/logits_resnet18'
    nModelos = args.nModelos

    workers = (int)(os.popen('nproc').read())
    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)
    test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)
    #almacenamos targets del dataset
    targets = []
    for x, t in test_loader:
        targets.append(np.array(t[0]))
    targets = torch.from_numpy(np.array(targets))
    logitsSof = []
    logits = []
    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))
        model.eval()
        logitSof, logit = explotation(model, test_loader, n, LOGITSPATH) 
        logits.append(logit)
        logitsSof.append(logitSof)

    logits = torch.Tensor(np.array(logits))
    avgACC = avgEnsemble(logitsSof, test_loader)

    print("Ensemble de {} modelos: {:.3f}".format(nModelos, 100*avgACC))

    n=0
    #calibracion de modelos individuales
    for logit in logits:
        print(logit.size())
        print(targets.size())
        calibracion(logit[-1], n, targets)
        n+=1


    
        
