from codigo.CIFAR10.resnet18 import PATH
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


def parse_args():
    parser = argparse.ArgumentParser(description='Parametros para configuracion del entrenamiento de las redes neuronales convolucionales')
    parser.add_argument('--nModelos', help="número de modelos que componen el emsemble", required=True, type = int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    PATH = './checkpoint'+'_resnet18'
    args = parse_args()
    nModelos = args.nModelos
    for n in range(nModelos):
        model = ResNet18()
        PATH = PATH + "_"+str(n+1) + '.pt'
        model.load_state_dict(torch.load(PATH))
        print("Modelo {} cargado correctamente".format(n+1))