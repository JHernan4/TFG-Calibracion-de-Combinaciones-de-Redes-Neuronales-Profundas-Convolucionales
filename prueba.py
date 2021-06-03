import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,250,250)
y = np.empty(250)
z = np.empty(250)
plt.figure()
plt.plot(x,y, label='1')
plt.plot(x,z, label='2')
plt.xlabel("Number of epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Resnet18 on CIFAR10")
plt.legend()
plt.savefig("Save Plot as PDF file using savefig.jpg")
