from typing import Counter
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


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = nn.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece





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
    Softmax = nn.Softmax(dim=1)
    softmaxes = [] #almacena los logits pasados por la Softmax
    with torch.no_grad():
        correct,total=0,0
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            logits=model.forward(x)
            softmax = Softmax(logits)
            softmaxes.append(Softmax) #meter esto en la funcion de calibracion
            index = torch.argmax(softmax, 1)
            total+=t.size(0)
            correct+=(t==index).sum().float()
    
    print("Modelo {}: accuracy {:.3f}".format(n+1, 100*(correct/total)))
    
    
    return softmaxes


def CalculaCalibracion(logits,targets, n):
    ECE,MCE,BRIER,NNL = 0.0,0.0,0.0,0.0
    counter = 0

    for logit, target in zip(logits, targets):
        calibrationMeasures = [compute_calibration_measures(logits, targets, False, 100)] 
        ECE,MCE,BRIER,NNL = ECE+calibrationMeasures[0],MCE+calibrationMeasures[1],BRIER+calibrationMeasures[2],NNL+calibrationMeasures[3]
        counter+=1

    print("Medidas de calibracion modelo {}: \n\tECE: {:.2f}%\n\tMCE: {:.2f}%\n\tBRIER: {:.2f}\n\tNNL: {:.2f}".format(n+1, 100*(ECE/counter), 100*(MCE/counter), BRIER/counter, NNL/counter))

    


def avgEnsemble(logits, testLoader, nModelos):
    avgLogits = []
    for i in range(len(logits[0])):
        avgLogits.append(logits[0][i]/nModelos)
    
    if nModelos > 1:
        for n in range(1, nModelos):
            for i in range(len(logits[n])):
                avgLogits[i]+=logits[n][i]/nModelos
    
    with torch.no_grad():
        correct,total=0,0
        for x,t in testLoader:
            x,t=x.cuda(),t.cuda()
            total+=t.size(0)
            index=torch.argmax(avgLogits[i],1)
            correct+=(t==index.cuda()).sum().float()
 
    
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
    
    softmaxes = []
    for n in range(nModelos):
        model = ResNet18()
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model.load_state_dict(torch.load(PATH+"_"+str(n+1) + '.pt'))
        print("Modelo {} cargado correctamente".format(n+1))
        model.eval()
        softmaxes = explotation(model, test_loader, n, LOGITSPATH) 
        

    avgACC, avgCalibracion = avgEnsemble(softmaxes, test_loader, nModelos)

    print("Ensemble de {} modelos: {:.3f}".format(nModelos, 100*avgACC))
    



    
        
