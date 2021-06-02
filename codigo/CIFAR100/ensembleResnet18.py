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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

#recibe un dataLoader y un modelo y devuelve los logits y el accuracy
def test(model, dataLoader):
    logits = torch.Tensor()
    total, correct = 0,0
    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        for x,t in dataLoader:
            x,t= x.cuda(), t.cuda()
            pred = model.forward(x)
            logit = sm(pred)
            index = torch.argmax(logit, 1)
            logits = torch.cat((logits, pred.cpu()), 0)
            total+=t.size(0)
            correct+=(t==index).sum().float()
        
    return logits, correct/total

 

#genera los logits promedio del ensemble
def generaLogitsPromedio(logitsModelos):
    avgLogits = logitsModelos[0]/len(logitsModelos)
    
    for n in range(1, len(logitsModelos)):
        avgLogits+=logitsModelos[n]/len(logitsModelos)

    return avgLogits


#calcula el % de accuracy dados unos logits y labels
def calculaAcuracy(logits, labels, batch_size=100):
    sm = nn.Softmax(dim=1)
    correct, total=0,0
    list_logits = torch.chunk(logits, batch_size)
    labels_list = torch.chunk(labels, batch_size)
    for logit, t in zip(list_logits, labels_list):
        logit = sm(logit)
        index = torch.argmax(logit, 1)
        total+=t.size(0)
        correct+=(t==index).sum().float()


    return correct/total


#dados logits y labels, calcula ECE, MCE, BRIER y NNL
def CalculaCalibracion(logits,labels):
    return compute_calibration_measures(logits, labels, False, 100)
    
        

#realiza la operacion Temp Scal (multiplica los logits recibidos por el parametro T)
def T_scaling(logits, t):
    return torch.mul(logits, t)
    
#recibe modelo y conjunto de validacion. Crea y optimiza el parametro T para usar en T_scaling
def temperatureScaling(model, validationLoader):
    temperature = nn.Parameter(torch.ones(1).cuda())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=20000)

    logits_list = torch.Tensor().cuda()
    labels_list = torch.LongTensor().cuda()
    for x,t in validationLoader:
        x,t = x.cuda(), t.cuda()
        model.eval()
        with torch.no_grad():
            logits_list = torch.cat((logits_list, model.forward(x)), 0)
            labels_list = torch.cat((labels_list, t), 0)
    
    def _eval():
        loss = criterion(T_scaling(logits_list, temperature), labels_list)
        loss.backward()
        return loss
    
    optimizer.step(_eval)
    cost = _eval().data #recuperamos el loss minimizado
    print("Final T_scaling factor con LBFGS: {:.2f}".format(temperature.item()))

    
    optimizer = torch.optim.SGD([temperature], lr=0.001, momentum=0.9)

    for e in range(20000):
        optimizer.zero_grad()
        loss = criterion(T_scaling(logits_list, temperature), labels_list)
        if loss <= cost:
            print("¡Convergencia conseguida!")
            break
        loss.backward()
        optimizer.step()

    print("Final T_scaling factor con SGD: {:.2f}".format(temperature.item()))
    return temperature.cpu()

def calc_bins(logits, labels, batch_size=100):
    sm = nn.Softmax(dim=1)
    list_preds = torch.chunk(logits, batch_size)
    list_labels = torch.chunk(labels, batch_size)
    preds = []
    labels_oneh = []
    for logit, label in zip(list_preds, list_labels):
        pred = sm(logit)
        pred = pred.cpu().detach().numpy()
        label_oneh = torch.nn.functional.one_hot(label, num_classes=10)
        label_oneh = label_oneh.cpu().detach().numpy()
        preds.extend(pred)
        labels_oneh.extend(label_oneh)
    
    preds = np.array(preds).flatten()
    labels_oneh = np.array(labels_oneh).flatten()
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

def get_metrics(preds, labels):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE

def draw_reliability_graph(preds, labels, file):
    ECE, MCE = get_metrics(preds, labels)
    bins, _, bin_accs, _, _ = calc_bins(preds, labels)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    #plt.show()
  
    plt.savefig(file, bbox_inches='tight')

    #draw_reliability_graph(preds)
   

if __name__ == '__main__':

    softmax = nn.Softmax(dim=1)

    testSize=8000 #tamanio del conjunto de test 
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
    test_loader, validation_loader = separarDataset(cifar10_test, testSize)

    #almacena las etiquetas del conjunto de test
    test_labels = torch.LongTensor()
    for x,t in test_loader:
        test_labels = torch.cat((test_labels, t), 0)

    modelos = [] #almacena los modelos leidos de cada fichero .pt
    logitsModelos = [] #lista que almacena los logits de todos los modelos
    logitsCalibrados = []
    for n in range(nModelos):
        model = ResNet18(num_classes=100)
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        modelos.append(model)
        print("Modelo {} cargado correctamente".format(n+1))
        logits, acc = test(model, test_loader)
        logitsModelos.append(logits)
        print("Accuracy modelo {}: {:.3f}".format(n+1, 100*acc))
        ECE, MCE, BRIER, NNL = CalculaCalibracion(softmax(logits), test_labels)
        print("Medidas de calibracion para el modelo {}:".format(n+1))
        print("\tECE: {:.2f}%\n\tMCE: {:.2f}%\n\tBRIER: {:.2f}\n\tNLL: {:.2f}".format(100*ECE, 100*MCE, BRIER, NNL))
        print("==> Aplicando Temp Scaling...")
        temperature = temperatureScaling(model, validation_loader)
        logitsCal = T_scaling(logits, temperature)
        logitsCalibrados.append(logitsCal)
        ECE, MCE, BRIER, NNL = CalculaCalibracion(softmax(logitsCal), test_labels)
        print("Medidas de calibracion para el modelo {}:".format(n+1))
        print("\tECE: {:.2f}%\n\tMCE: {:.2f}%\n\tBRIER: {:.2f}\n\tNLL: {:.2f}".format(100*ECE, 100*MCE, BRIER, NNL))

    print("Medidas para el ensemble de {} modelos".format(nModelos))
    avgLogits = generaLogitsPromedio(logitsModelos)
    print("\tAccuracy: {:.2f}".format(100*calculaAcuracy(avgLogits, test_labels)))
    ECE, MCE, BRIER, NNL = CalculaCalibracion(softmax(avgLogits), test_labels)
    print("\tECE: {:.2f}%\n\tMCE: {:.2f}%\n\tBRIER: {:.2f}\n\tNLL: {:.2f}".format(100*ECE, 100*MCE, BRIER, NNL))
    print("==> Aplicando Temp Scaling al ensemble")
    avgLogitsCalibrados = generaLogitsPromedio(logitsCalibrados)
    ECE, MCE, BRIER, NNL = CalculaCalibracion(softmax(T_scaling(avgLogitsCalibrados, temperature)), test_labels)
    print("\tECE: {:.2f}%\n\tMCE: {:.2f}%\n\tBRIER: {:.2f}\n\tNLL: {:.2f}".format(100*ECE, 100*MCE, BRIER, NNL))