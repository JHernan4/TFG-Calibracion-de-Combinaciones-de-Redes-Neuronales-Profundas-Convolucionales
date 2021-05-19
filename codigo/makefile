#ejecutar resnet18 para generacion de modelos
CIFAR10resnet18: CIFAR10/resnet18.py
	python3 CIFAR10/resnet18.py --seed 123 --nEpocas 250 --nModelos 5

#ejecutar resnet50 para generacion de modelos
CIFAR10resnet50: CIFAR10/resnet50.py
	python3 CIFAR10/resnet50.py --seed 123 --nEpocas 250 --nModelos 5

CIFAR10densenet121: CIFAR10/densenet121.py
	python3 CIFAR10/densenet121.py --seed 123 --nEpocas 250 --nModelos 5

CIFAR10ensembleResnet18: CIFAR10/ensembleResnet18.py
	python3 CIFAR10/ensembleResnet18.py --nModelos 5

CIFAR10ensembleResnet50: CIFAR10/ensembleResnet50.py
	python3 CIFAR10/ensembleResnet50.py --nModelos 5

CIFAR10ensembleDensenet121: CIFAR10/ensembleDensenet121.py
	python3 CIFAR10/ensembleDensenet121.py --nModelos 5

CIFAR10comprimirCheckpoints:
	tar -zcvf checkpointResnet18.tar.gz checkpointResnet18
	tar -zcvf checkpointResnet50.tar.gz checkpointResnet50
	tar -zcvf checkpointDensenet121.tar.gz checkpointDensenet121