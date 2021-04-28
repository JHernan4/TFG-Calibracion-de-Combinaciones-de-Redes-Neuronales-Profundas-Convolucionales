import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import time

CUDA = torch.cuda.is_available()

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)#capa de salida
        self.loss_criterion = nn.CrossEntropyLoss()#Función de pérdida
        
    def forward(self, x, target):
        x = x.view(-1, 1, 28, 28)#transforma las imágenes a tamaño (n, 1, 28, 28)
        x = F.relu(self.conv1(x))
        # la salida de conv1 es 32x26x26
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)#El 2 es el stride
        # la salida de conv2 es 64x24x24, la salida de max_pool es 64x12x12
        x = x.view(-1, 64*12*12)#transformamos la salida en un tensor de tamaño (n, 9216)
        x = F.relu(self.fc1(x))#Función de activación relu en la salida de la capa oculta
        x = F.softmax(self.fc2(x), dim=1)#Función de activación softmax en la salida de la capa oculta
        return x
        

class NN():
    """docstring for NN"""
    def __init__(self):
        self.model = NET()
        if CUDA:
            self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def epoch_step(self, dataset_loader, train=False, verbose=2):
        correct_cnt, ave_loss = 0, 0#Contador de aciertos y acumulador de la función de pérdida
        count = 0#Contador de muestras
        for batch_idx, (x, target) in enumerate(dataset_loader):
            start_time_epch = time.time()
            count += len(x)#sumamos el tamaño de batch, esto es porque n_batches*tamaño_batch != n_muestras
            if train:
                self.optimizer.zero_grad()#iniciamos a 0 los valores de los gradiente
            x, target = Variable(x), Variable(target)#Convertimos el tensor a variable del modulo autograd
            if CUDA:
                x = x.cuda()
                target = target.cuda()
            score = self.model(x, target)#realizamos el forward
            loss = self.model.loss_criterion(score, target)
            _, pred_label = torch.max(score.data, 1)#pasamos de one hot a número
            correct_cnt_epch = (pred_label == target.data).sum()#calculamos el número de etiquetas correctas
            correct_cnt += correct_cnt_epch
            ave_loss += loss.data#sumamos el resultado de la función de pérdida para mostrar después
            if train:
                loss.backward()#Calcula los gradientes y los propaga 
                self.optimizer.step()#adaptamos los pesos con los gradientes propagados
            elapsed_time_epch = time.time() - start_time_epch

            if verbose == 1:
                elapsed_time_epch = time.strftime("%Hh,%Mm,%Ss", time.gmtime(elapsed_time_epch))
                print ('\t\tbatch: {} loss: {:.6f}, accuracy: {:.4f}, time: {}'.format(
                    batch_idx, loss.data[0], correct_cnt_epch/len(x), elapsed_time_epch))

        accuracy = correct_cnt/count#Calculamos la precisión total
        ave_loss /= count#Calculamos la pérdida media

        return ave_loss, accuracy

    def train(self, epoch, train_loader, test_loader=None, verbose=2):
        for epoch in range(epoch):
            if verbose > 0:
                print("\n***Epoch {}***\n".format(epoch))
                if verbose == 1:
                    print("\tTraining:")
            start_time = time.time()
            ave_loss, accuracy = self.epoch_step(train_loader, train=True, verbose=verbose)
            elapsed_time = time.time() - start_time
            if verbose == 2:
                elapsed_time = time.strftime("%Hh,%Mm,%Ss", time.gmtime(elapsed_time))
                print ('\tTraining loss: {:.6f}, accuracy: {:.4f}, time: {}'.format(
                    ave_loss, accuracy, elapsed_time))

            if test_loader != None:
                start_time = time.time()
                ave_loss, accuracy = self.epoch_step(test_loader, train=False, verbose=verbose)
                elapsed_time = time.time() - start_time
                if verbose == 2:
                    elapsed_time = time.strftime("%Hh,%Mm,%Ss", time.gmtime(elapsed_time))
                    print ('\tTest loss: {:.6f}, accuracy: {:.4f}, time: {}'.format(
                        ave_loss, accuracy, elapsed_time))

    def evaluate(self, test_loader, verbose=2):
        print("\n***Evaluate***\n")
        start_time = time.time()
        ave_loss, accuracy = self.epoch_step(test_loader, train=False, verbose=verbose)
        elapsed_time = time.time() - start_time
        if verbose == 2:
            elapsed_time = time.strftime("%Hh,%Mm,%Ss", time.gmtime(elapsed_time))
            print ('\tloss: {:.6f}, accuracy: {:.4f}, time: {}'.format(
                ave_loss, accuracy, elapsed_time))

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    torch.manual_seed(123) #fijamos la semilla
    epochs = 10

    trans = transforms.Compose([transforms.ToTensor()]) #Transformador para el dataset
    root="../data"
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)

    print ('Trainning batch number: {}'.format(len(train_loader)))
    print ('Testing batch number: {}'.format(len(test_loader)))

    net = NN()
    net.train(epochs, train_loader, test_loader=test_loader, verbose=2)
    net.evaluate(test_loader, verbose=2)