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
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description='Parametros para configuracion del entrenamiento de las redes neuronales convolucionales')
    parser.add_argument('--nModelos', help="número de modelos que componen el emsemble", required=True, type = int)
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
    worker_seed=torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def explotation(model, testLoader, n):
    softmax = nn.Softmax(dim=1)
    logits = []
    targets = []
    with torch.no_grad():
        correct,total=0,0
        i=0
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            test_pred=model.forward(x)
            logit=softmax(test_pred).cpu()
            logits.append(torch.argmax(logit,1))
            index=torch.argmax(logit,1)
            total+=t.size(0)
            correct+=accuracy_score(t.cpu(), index, normalize=False)
    
    print("Modelo {}: accuracy {:.3f}".format(n+1, 100*(correct/total)))
    return logits


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
            correct+=accuracy_score(t, avgLogits[i].cuda(), normalize=False)
            i=i+1

    return correct/total

if __name__ == '__main__':
    args = parse_args()
    PATH = './checkpoint'+'_resnet18'
    nModelos = args.nModelos

    workers = (int)(os.popen('nproc').read())
    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=False,transform=cifar10_transforms_test)
    test_loader = torch.utils.data.DataLoader(cifar10_test,batch_size=100,shuffle=False,num_workers=workers)

    logits = []
    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))
        model.eval()
        logits.append(explotation(model, test_loader, n))

    avgACC = avgEnsemble(logits, test_loader)

    print("Ensemble de {} modelos: {:.3f}".format(nModelos, 100*avgACC))

    
        
