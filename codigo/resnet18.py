import torch
if not torch.cuda.is_available():
	print("unable to run on GPU")
	exit(-1)
import torchvision #computer vision dataset module
import torchvision.models as models
from torchvision import datasets,transforms
from torch import nn

import numpy
import os


def lr_scheduler(epoch):
	if epoch < 150:
		return 0.1
	elif epoch < 250:
		return 0.01
	elif epoch < 350:
		return 0.001

if __name__ == '__main__':

	#definimos el modelo de red neuronal y lo movemos a la GPU
	resnet18 = models.resnet18(True)
	print(resnet18)
