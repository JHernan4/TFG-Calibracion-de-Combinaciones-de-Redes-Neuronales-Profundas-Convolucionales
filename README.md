# TFG-Calibracion de Combinaciones de Redes Neuronales Profundas Convolucionales

Git repository corresponding to the research work done by me for my Final Degree Thesis for Informatics Engineering. The project sum up the research about the unusual behaivour 
when differents models of convolutional neuroanl networks are combined (it's konwn as ensembles). 

For carry through the research, I've used Python programming languague and his library Pytorch, perfect to work with neural networks. The tests are too hard computacionally. It is therefore that I needed several GPU units. The code is prepared for be runned on GPU and CPU units

This repository is organized in two main folders:

* **Codigo**: in this folder we can find all python scripts necesaries. They are organized according to dataset processed and model of neural network that is used. For example in *CIFAR10/resnet18.py* we can find the script which train some neural networks *ResNet18*. Into this folders (CIFAR10 and CIFAR100) we can find other folders with name CheckPoint... Here are stored *.pt* extension files, which are special files to save the trained models. All scripts contains a help menu. Only you must execute the command: *python3 script.py -h*. You will see on the screen the arguments that the script need to run correctly. +++


* **Memoria**: this folder only contains one file whit the final pdf which was exhibited to the magistrates. 


+++ Also, install some external Python libraries must be installed. For example torch, torchvision, numpy, scipy...
