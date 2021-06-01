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

#recibe el dataset y devuelve dos conjuntos, uno de test (90%) y otro de validacion (10%)
def separarDataset(dataset, porc_test=0.9, porc_val=0.1):
    val_set, test_set = torch.utils.data.random_split(cifar10_test, [len(dataset)*porc_val, len(dataset)*porc_test])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=workers)

    return test_loader, val_loader

#recibe un dataLoader y un modelo y devuelve los logits y targets
def procesarConjunto(model, dataLoader):
    logits = []
    labels = []

    for x,t in dataLoader:
        x,t= x.cuda(), t.cuda()
        labels.append(t)
        logits.append(model.forward(x))
    
    return torch.cat(logits).cuda(), torch.cat(labels).cuda()

def test(model, dataLoader, nClases=10):
    sm = nn.Softmax(dim=1)
    preds = []
    labels_oneh = []
    correct = 0
    counter = 0

    model.eval()
    with torch.no_grad():
        for x, t in dataLoader:
            x,t = x.cuda(), t.cuda()
            pred = model.forward(x)
            pred = sm(pred)

            _, predicted_cl = torch.max(pred.data, 1)
            pred = pred.cpu().detach().numpy()

            label_oneh = nn.functional.one_hot(t, num_classes=nClases)
            label_oneh  = label_oneh.cpu().detach().numpy()

            preds.extend(pred)
            labels_oneh.extend(label_oneh)

            correct+= sum(predicted_cl == t).item()
            
            counter+=t.size(0)

    preds = np.array(preds).flatten()
    labels_oneh = np.array(labels_oneh).flatten()

    return preds, label_oneh, correct/counter    

#recibe el modelo y el conjunto de validacion y devuelve los logits pasados por la softmax
def procesaValidacion(model, valLoader):
    Softmax = nn.Softmax(dim=1)
    logitsS = []
    softmaxes = []
    with torch.no_grad():
        for x,t in valLoader:
            x,t=x.cuda(),t.cuda()
            logits=model.forward(x)
            softmax = Softmax(logits)
            logitsS.append(np.array(logits.cpu())) #meter esto en la funcion de calibracion
            softmaxes.append(np.array(softmax.cpu())) #meter esto en la funcion de calibracion
    
    return torch.Tensor(np.array(logitsS))

#genera los logits del conjunto de test y los devuelve pasados por la softmax
def generarLogits(model, testLoader):
    Softmax = nn.Softmax(dim=1)
    logitsS = []
    softmaxes = []
    with torch.no_grad():
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            logits=model.forward(x)
            softmax = Softmax(logits)
            logitsS.append(np.array(logits.cpu())) #meter esto en la funcion de calibracion
            softmaxes.append(np.array(softmax.cpu())) #meter esto en la funcion de calibracion
    
    return torch.Tensor(np.array(logitsS)), torch.Tensor(np.array(softmaxes))

#calcula el % de accuracy dados unos logits y labels
def calculaAcuracy(logits, labels):
    total, correct = 0,0
    for logit, t in zip(logits, labels):
        index = torch.argmax(logit, 1)
        total+=t.size(0)
        correct+=(t==index).sum().float()
    
    return correct/total

#calcula el accuracy de un ensemble de modelos dados los logits de todos los modelos y los labels 
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

#dados logits y labels, calcula ECE, MCE, BRIER y NNL
def CalculaCalibracion(logits,labels):
    ECE,MCE,BRIER,NNL = 0.0,0.0,0.0,0.0
    counter = 0
    for logit, label in zip(logits, labels):
        calibrationMeasures = [compute_calibration_measures(logit, label, False, 100)]
        ECE,MCE,BRIER,NNL = ECE+calibrationMeasures[0][0],MCE+calibrationMeasures[0][1],BRIER+calibrationMeasures[0][2],NNL+calibrationMeasures[0][3]
        counter+=1
    return [ECE/counter, MCE/counter, BRIER/counter, NNL/counter]


#crea y optimiza un parametro T para el Temp Scal con el CONJUNTO DE VALIDACION
def entrenaParametroT(logitsVal, labelsVal):
    temperature = nn.Parameter(torch.ones(1) * 0.1)
    loss = nn.CrossEntropyLoss()

    for e in range(2000):
        optimizer = torch.optim.SGD([temperature], lr=0.001, momentum=0.9)
        for logit, label in zip(logits, labels):
            cost = loss(logit * temperature, label)
            cost.backward()
            optimizer.step(eval)
            optimizer.zero_grad()
    print('Optimal temperature: %.3f' % temperature.item())
    return temperature


def T_scaling(logits, t):
    return torch.mul(logits, t)
    
#realiza el Temp Scal sobre el CONJUNTO DE TEST
def temperatureScaling(model, validationLoader):
    temperature = nn.Parameter(torch.ones(1).cuda())
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=2000)

    logits_list = []
    labels_list = []
    temps = []
    losses = []

    for x,t in validationLoader:
        x,t = x.cuda(), t.cuda()
        model.eval()

        with torch.not_grad():
            logits_list.append(model.forward(x))
            labels_list.append(t)

    #creamos los tensores
    logits_list = torch.cat(logits_list).cuda()
    labels_list = torch.cat(labels_list).cuda()

    def _eval():
        cost = loss(T_scaling(logits_list, temperature), labels_list)
        cost.backward()
        temps.append(temperature.item())
        losses.append(cost)
        return cost

    optimizer.step(_eval)

    print("Final T_scaling factor: {:.2f}".format(t.item()))

    return temperature
        
   

if __name__ == '__main__':
    testSize=9000
    args = parse_args()
    PATH = './checkpointResnet18/checkpoint_resnet18'
    nModelos = args.nModelos

    workers = (int)(os.popen('nproc').read())
    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=True,transform=cifar10_transforms_test)

    test_loader, validation_loader = separarDataset(cifar10_test)    

    labels=[]
    for x,t in test_loader: 
        labels.append(t)
    
    labelsVal = []
    for x, t in validation_loader:
        labelsVal.append(t)

    softmaxes = []
    logitsS = []
    softmaxesVal = []
    modelos = []
    '''
    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))
        modelos.append(model)
        model.eval()
        logitsAux, logits = generarLogits(model, test_loader)
        softmaxesVal.append(procesaValidacion(model, val_loader))
        softmaxes.append(logits)
        logitsS.append(logitsAux)
        acc = calculaAcuracy(logits, labels)
        print("Accuracy modelo {}: {:.3f}".format(n+1, 100*acc))
        medidasCalibracion = CalculaCalibracion(logits, labels)
        print("Medidas de calibracion modelo {}: \n\tECE: {:.3f}%\n\tMCE: {:.3f}%\n\tBRIER: {:.3f}\n\tNNL: {:.3f}".format(n+1, 100*(medidasCalibracion[0]), 100*(medidasCalibracion[1]), medidasCalibracion[2], medidasCalibracion[3]))
        

    accEnsemble, avgLogits = accuracyEnsemble(softmaxes, labels)
    print("Accuracy del ensemble de {} modelos: {:.3f}".format(nModelos, 100*accEnsemble))
    medidasCalibracionEnsemble = CalculaCalibracion(avgLogits, labels)
    print("Medidas de calibracion ensemble {} modelos: \n\tECE: {:.3f}%\n\tMCE: {:.3f}%\n\tBRIER: {:.3f}\n\tNNL: {:.3f}".format(nModelos, 100*(medidasCalibracionEnsemble[0]), 100*(medidasCalibracionEnsemble[1]), medidasCalibracionEnsemble[2], medidasCalibracionEnsemble[3]))
    
    print("==> Aplicando temperature scaling")

    
    for n, model in enumerate(modelos):
        t = temperatureScaling(model, val_loader)



    '''
    
    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))

        logits, labels, acc = test(model, test_loader)
        print("Accuracy modelo {}: {}".format(n+1, 100*acc))        
