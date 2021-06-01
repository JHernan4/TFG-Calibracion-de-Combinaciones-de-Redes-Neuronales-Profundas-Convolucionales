import torch
if not torch.cuda.is_available():
    print("Error al cargar GPU")
    exit(-1)
torch.manual_seed(123)
import torchvision.models as models
from torchvision import datasets,transforms
from torch import nn
import sys
sys.path.append("../models")
sys.path.append("../calibration")
from resnet import ResNet18
from utils_calibration import compute_calibration_measures
import numpy as np
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
def separarDataset(dataset, testSize=9000):
    val_set, test_set = torch.utils.data.random_split(cifar10_test, [len(dataset)-testSize, testSize])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=workers)

    return test_loader, val_loader

#recibe un dataLoader y un modelo y devuelve los logits y targets
def test(model, dataLoader):
    logits = []

    for x,t in dataLoader:
        x,t= x.cuda(), t.cuda()
        pred = model.forward(x)
        logits.append(pred.cpu().detach().numpy())
        
    logits = np.array(logits)
    return torch.from_numpy(logits)

#genera los logits promedio del ensemble
def generaLogitsPromedio(logitsModelos):
    avgLogits = []
    for i in range(len(logitsModelos[0])):
        avgLogits.append(logitsModelos[0][i]/len(logitsModelos))
    
    for n in range(1, len(logitsModelos)):
        for i in range(len(logitsModelos[n])):
            avgLogits[i]+=logitsModelos[n][i]/len(logitsModelos)
    
    return avgLogits

#calcula el % de accuracy dados unos logits y labels
def calculaAcuracy(logits, labels):
    total, correct = 0,0
    sm = nn.Softmax(dim=1)
    for logit, t in zip(logits, labels):
        logit = sm(logit)
        index = torch.argmax(logit, 1)
        total+=t.size(0)
        correct+=(t==index).sum().float()
    
    return correct/total


#dados logits y labels, calcula ECE, MCE, BRIER y NNL
def CalculaCalibracion(logits,labels):
    sm = nn.Softmax(dim=1)
    ECE,MCE,BRIER,NNL = 0.0,0.0,0.0,0.0
    counter = 0
    for logit, label in zip(logits, labels):
        logit = sm(logit)
        calibrationMeasures = [compute_calibration_measures(logit, label, False, 100)]
        ECE,MCE,BRIER,NNL = ECE+calibrationMeasures[0][0],MCE+calibrationMeasures[0][1],BRIER+calibrationMeasures[0][2],NNL+calibrationMeasures[0][3]
        counter+=1
    return [ECE/counter, MCE/counter, BRIER/counter, NNL/counter]

#realiza la operacion Temp Scal (multiplica los logits recibidos por el parametro T)
def T_scaling(logits, t):
    return torch.mul(logits, t)
    
#recibe modelo y conjunto de validacion. Crea y optimiza el parametro T para usar en T_scaling
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

        with torch.no_grad():
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

    print("Final T_scaling factor: {:.2f}".format(temperature.item()))

    return temperature
        
   

if __name__ == '__main__':
    testSize=9000 #tamanio del conjunto de test 
    args = parse_args()
    PATH = './checkpointResnet18/checkpoint_resnet18' #ruta para lectura de los checkpoints de los modelos
    nModelos = args.nModelos

    workers = (int)(os.popen('nproc').read())

    #transformaciones que se aplican al dataset
    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    #descarga del dataset y aplicacion de las transformaciones
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=True,transform=cifar10_transforms_test)

    #separa dataset en TEST y VALIDACION
    test_loader, validation_loader = separarDataset(cifar10_test)    

    #almacena las etiquetas del conjunto de validacion
    validation_labels = []
    for x,t in validation_loader:
        validation_labels.append(t)

    #almacena las etiquetas del conjunto de test
    test_labels = []
    for x,t in test_loader:
        test_labels.append(t)
    
    modelos = [] #almacena los modelos leidos de cada fichero .pt
    logitsModelos = [] #lista que almacena los logits de todos los modelos

    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        modelos.append(model)
        print("Modelo {} cargado correctamente".format(n+1))
        logits = test(model, test_loader)
        logitsModelos.append(logits)
        acc = calculaAcuracy(logits, test_labels)
        print("Accuracy modelo {}: {:.3f}".format(n+1, 100*acc))
    
    avgLogits = generaLogitsPromedio(logitsModelos)
    accEsemble = calculaAcuracy(avgLogits, test_labels)
    print("Accuracy ensemble de {} modelos: {:.3f}".format(n+1, 100*accEsemble))

    for n, model in enumerate(modelos):
        medidasCalibracion = CalculaCalibracion(logitsModelos[n], test_labels)
        print("Medidas de calibracion modelo {}: \n\tECE: {:.3f}%\n\tMCE: {:.3f}%\n\tBRIER: {:.3f}\n\tNNL: {:.3f}".format(n+1, 100*(medidasCalibracion[0]), 100*(medidasCalibracion[1]), medidasCalibracion[2], medidasCalibracion[3]))
        print("Aplicando Temp Scal...")
        temperature = temperatureScaling(model, validation_loader)
        medidasCalibracion = CalculaCalibracion(T_scaling(logitsModelos[n], temperature), test_labels)
        print("Medidas de calibracion modelo {}: \n\tECE: {:.3f}%\n\tMCE: {:.3f}%\n\tBRIER: {:.3f}\n\tNNL: {:.3f}".format(n+1, 100*(medidasCalibracion[0]), 100*(medidasCalibracion[1]), medidasCalibracion[2], medidasCalibracion[3]))
        
