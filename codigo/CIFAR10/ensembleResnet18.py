import torch
if not torch.cuda.is_available():
    print("Error al cargar GPU")
    exit(-1)
torch.manual_seed(123)
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
def separarDataset(dataset, testSize=9000):
    val_set, test_set = torch.utils.data.random_split(cifar10_test, [len(dataset)-testSize, testSize])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=workers)

    return test_loader, val_loader

#recibe un dataLoader y un modelo y devuelve los logits y targets
def test2(model, dataLoader):
    logits = []

    for x,t in dataLoader:
        x,t= x.cuda(), t.cuda()
        pred = model.forward(x)
        logits.append(pred.cpu().detach().numpy())
        
    logits = np.array(logits)
    return torch.from_numpy(logits)

def test(model, dataLoader, nClases=10, calibracion=False, temperature=None):
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
            
            if calibracion == True and temperature is not None:
                pred = T_scaling(pred, temperature)

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

    return preds, labels_oneh, correct/counter    

def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels_oneh):
  ECE = 0
  MCE = 0
  loss = nn.CrossEntropyLoss()
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)
    NLL = loss(preds, labels_oneh)


  return ECE, MCE, NLL


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
    testSize=9000
    args = parse_args()
    PATH = './checkpointResnet18/checkpoint_resnet18'
    nModelos = args.nModelos

    workers = (int)(os.popen('nproc').read())
    cifar10_transforms_test=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    cifar10_test=datasets.CIFAR10('/tmp/',train=False,download=True,transform=cifar10_transforms_test)

    test_loader, validation_loader = separarDataset(cifar10_test)    

    validation_labels = []
    for x,t in validation_loader:
        validation_labels.append(t)

    test_labels = []
    for x,t in test_loader:
        test_labels.append(t)
    

    softmaxes = []
    logitsS = []
    softmaxesVal = []
    modelos = []
    logitsModelos = []

    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))
        logits = test2(model, test_loader)
        logitsModelos.append(logits)
        acc = calculaAcuracy(logits, test_labels)
        print("Accuracy modelo {}: {:.3f}".format(n+1, 100*acc))
    
    avgLogits = generaLogitsPromedio(logitsModelos)
    accEsemble = calculaAcuracy(avgLogits, test_labels)
    print("Accuracy ensemble de {} modelos: {:.3f}".format(n+1, 100*accEsemble))
    
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
    '''
    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))

        logits, labels, acc = test(model, test_loader)
        print("Accuracy modelo {}: {:.2f}".format(n+1, 100*acc))
        ECE, MCE, NLL= get_metrics(logits, labels)       
        print("ECE: {}%, MCE: {}%, NLL: {}".format(ECE*100, MCE*100, NLL))

        t = temperatureScaling(model, validation_loader)
        print("==> Aplicando temp scaling")
        logits, labels, acc = test(model, test_loader, 10, True, t)
        print("Accuracy modelo {}: {:.2f}".format(n+1, 100*acc))
        ECE, MCE, NLL= get_metrics(logits, labels)       
        print("ECE: {}%, MCE: {}%, NLL: {}".format(ECE*100, MCE*100, NLL))
    '''