import torch
if not torch.cuda.is_available():
    print("Error al cargar GPU")
    exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
from torch.utils.data.sampler import SubsetRandomSampler
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
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
    worker_seed=torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generarLogitsValidacion(model, valLoader):
    Softmax = nn.Softmax(dim=1)
    softmaxes = [] 
    with torch.no_grad():
        for x,t in valLoader:
            x,t=x.cuda(),t.cuda()
            logits=model.forward(x)
            softmax = Softmax(logits)
            softmaxes.append(np.array(softmax.cpu())) #meter esto en la funcion de calibracion
    
    return torch.Tensor(np.array(softmaxes))

def generarLogits(model, testLoader):
    Softmax = nn.Softmax(dim=1)
    softmaxes = [] 
    with torch.no_grad():
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            logits=model.forward(x)
            softmax = Softmax(logits)
            softmaxes.append(np.array(softmax.cpu())) #meter esto en la funcion de calibracion
    
    return torch.Tensor(np.array(softmaxes))


def calculaAcuracy(logits, labels):
    total, correct = 0,0
    for logit, t in zip(logits, labels):
        index = torch.argmax(logit, 1)
        total+=t.size(0)
        correct+=(t==index).sum().float()
    
    return correct/total

def accuracyEnsemble(logits, labels):
    avgLogits = []
    for i in range(len(logits[0])):
        avgLogits.append(logits[0][i]/len(logits))
    
    for n in range(1, len(logits)):
        for i in range(len(logits[n])):
            avgLogits[i]+=logits[n][i]/len(logits)

    correct,total=0,0
    for avgLogit, t in zip(avgLogits, labels):
        total+=t.size(0)
        index=torch.argmax(avgLogit,1)
        correct+=(t==index).sum().float()

    return correct/total, avgLogits

def CalculaCalibracion(logits,labels):
    ECE,MCE,BRIER,NNL = 0.0,0.0,0.0,0.0
    counter = 0
    for logit, label in zip(logits, labels):
        calibrationMeasures = [compute_calibration_measures(logit, label, False, 100)]
        ECE,MCE,BRIER,NNL = ECE+calibrationMeasures[0][0],MCE+calibrationMeasures[0][1],BRIER+calibrationMeasures[0][2],NNL+calibrationMeasures[0][3]
        counter+=1
    return [ECE/counter, MCE/counter, BRIER/counter, NNL/counter]


def entrenaParametroT(logits, labels):
    temperature = nn.Parameter(torch.ones(100, 10) * 1.5)
    optimizer=torch.optim.SGD([temperature],lr=0.01)
    loss = nn.CrossEntropyLoss()
    def eval():
        for logit,label in zip (logits, labels):
            o = loss(temperature*logit, label)
            o.backward()
        return o
    optimizer.step(eval)

    return temperature
    

def tempScaling(logits, labels):
    
    temperature = entrenaParametroT(logits, labels)
    logitsTemp = []
    for logit in logits:
        logit = logit * temperature
        logitsTemp.append(logit.detach().numpy())
    return torch.Tensor(np.array(logitsTemp))
    

    

if __name__ == '__main__':
    testSize=9000
    args = parse_args()
    PATH = './checkpointResnet18/checkpoint_resnet18'
    nModelos = args.nModelos

    workers = (int)(os.popen('nproc').read())
    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=True,transform=cifar10_transforms_test)
    val_set, test_set = torch.utils.data.random_split(cifar10_test, [1000, 9000])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=workers)

    labels=[]
    for x,t in test_loader: 
        labels.append(t)
    
    softmaxes = []
    softmaxesVal = []
    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))
        model.eval()
        logits = generarLogits(model, test_loader)
        softmaxesVal.append(generarLogitsValidacion(model, val_loader))
        softmaxes.append(logits)
        acc = calculaAcuracy(logits, labels)
        print("Accuracy modelo {}: {:.3f}".format(n+1, 100*acc))
        medidasCalibracion = CalculaCalibracion(logits, labels)
        print("Medidas de calibracion modelo {}: \n\tECE: {:.3f}%\n\tMCE: {:.3f}%\n\tBRIER: {:.3f}\n\tNNL: {:.3f}".format(n+1, 100*(medidasCalibracion[0]), 100*(medidasCalibracion[1]), medidasCalibracion[2], medidasCalibracion[3]))
        

    accEnsemble, avgLogits = accuracyEnsemble(softmaxes, labels)
    print("Accuracy del ensemble de {} modelos: {:.3f}".format(nModelos, 100*accEnsemble))
    medidasCalibracionEnsemble = CalculaCalibracion(avgLogits, labels)
    print("Medidas de calibracion ensemble {} modelos: \n\tECE: {:.3f}%\n\tMCE: {:.3f}%\n\tBRIER: {:.3f}\n\tNNL: {:.3f}".format(nModelos, 100*(medidasCalibracionEnsemble[0]), 100*(medidasCalibracionEnsemble[1]), medidasCalibracionEnsemble[2], medidasCalibracionEnsemble[3]))
    
    print("==> Aplicando temp scaling")

    for logitsVal in softmaxesVal:
        logitsTemp = tempScaling(logitsVal, labels)
        medidasCalibracionTemp = CalculaCalibracion(logitsTemp, labels)
        print("Medidas de calibracion modelo {} con Temperature Scaling: \n\tECE: {:.3f}%\n\tMCE: {:.3f}%\n\tBRIER: {:.3f}\n\tNNL: {:.3f}".format(n+1, 100*(medidasCalibracionTemp[0]), 100*(medidasCalibracionTemp[1]), medidasCalibracionTemp[2], medidasCalibracionTemp[3]))




    
        
