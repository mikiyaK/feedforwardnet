#ffnn.py
from __future__ import print_function #for using python3's print_function in python2
import time
import random
import numpy as np
import scipy.stats as stats
import sys
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def cross_entropy(prob, t):
    return -np.mean(np.log(prob[np.arange(prob.shape[0]), t] + 1e-7))

def softmax(y):
    y = y - np.max(y, axis=1, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

def softmax_cross_entropy(y, t):
    return cross_entropy(softmax(y), t)

class Sequential_1:
    def __init__(self, layers = []):
        self.layers = layers
    def addlayer(self, layer):
        self.layers.append(layer)
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def update(self):
        for l in self.layers:
            l.update()

    def zerograd(self):
        for l in self.layers:
            l.zerograd()
class Sequential_2:
    def __init__(self, layers = []):
        self.layers = layers
    def addlayer(self, layer):
        self.layers.append(layer)
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def update(self):
        for l in self.layers:
            l.update()

    def zerograd(self):
        for l in self.layers:
            l.zerograd()
class Sequential_3:
    def __init__(self, layers = []):
        self.layers = layers
    def addlayer(self, layer):
        self.layers.append(layer)
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def update(self):
        for l in self.layers:
            l.update()

    def zerograd(self):
        for l in self.layers:
            l.zerograd()
class Sequential_4:
    def __init__(self, layers = []):
        self.layers = layers
    def addlayer(self, layer):
        self.layers.append(layer)
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def update(self):
        for l in self.layers:
            l.update()

    def zerograd(self):
        for l in self.layers:
            l.zerograd()


class Classifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x, t):
        y = self.model.forward(x)
        #pred = np.argmax(y, axis=1) #pred represents the prediction of the number for each image 
        #acc = 1.0 * np.where(pred == t)[0].size / y.shape[0]
        #loss = util.softmax_cross_entropy(y,t)
        return y

    def update(self, x, t):
        self.model.zerograd()
        y = self.model.forward(x)
        pred = np.argmax(y, axis=1)
        acc = 1.0 * np.where(pred == t)[0].size / y.shape[0] #devide the number of currect answer in the batch by batch size
        prob = softmax(y)#change output to probability (normalization)
        loss = cross_entropy(prob, t)#loss function
        dout = prob
        dout[np.arange(dout.shape[0]), t] -= 1 #differentiate loss function by y 
        self.model.backward(dout / dout.shape[0])#calculate partial differentiations by each parameters of each layer to use in next update() function to update parameters. 
        self.model.update()#update parameters based on the partial differntials
        return loss, acc

class Layer(object):
    def __init__(self, lr=0.01, momentum=0.90, weight_decay_rate=5e-4):
        self.params = {}
        self.grads = {} #partial differntials of current time
        self.v = None #the velue which reflects not only effect of current time partial differentials but also effect of past partial differentials
        self.momentum = momentum #the coefficient of inertia term
        self.lr = lr #learning rate
        self.weight_decay_rate = weight_decay_rate #something like attenuation rate for the nurm of each parameter

    def update(self):
        if self.v == None:
            self.v = {}
            for k in self.params.keys():
                self.v[k] = np.zeros(shape = self.params[k].shape, dtype = self.params[k].dtype)
        for k in self.params.keys():
            self.v[k] = self.v[k] * self.momentum - self.lr * self.grads[k]
            self.params[k] = (1 - self.lr * self.weight_decay_rate) * self.params[k] + self.v[k]

    def zerograd(self):
        for k in self.params.keys():
            self.grads[k] = np.zeros(shape = self.params[k].shape, dtype = self.params[k].dtype)


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.params['W'] = np.random.normal(scale=np.sqrt(1.0/input_dim), size=(input_dim, output_dim)).astype(np.float32) #use normal distribution for initializing 'W' 
        self.params['b'] = np.zeros(shape = (1, output_dim), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grads['W'] = np.dot(self.x.T,dout)
        self.grads['b'] = np.sum(dout,axis = 0,keepdims=True)
        return np.dot(dout,self.params['W'].T)

class ReLULayer(Layer):
    def __init__(self):
        super(ReLULayer, self).__init__()

    def forward(self, x):
        out = np.maximum(x, 0)
        self.mask = np.sign(out)
        return out

    def backward(self, dout):
        return self.mask * dout
    
class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        self.original_shape = x.shape
        return np.reshape(x, (x.shape[0], x.size // x.shape[0]))

    def backward(self, dout):
        return np.reshape(dout, self.original_shape)


def main(epoch):
    model_1 = Sequential_1()   #model composes the sequential of hidden layers of the neural network by using Sequential() 
    model_1.addlayer(FlattenLayer())
    model_1.addlayer(LinearLayer(784,20))
    #model_1.addlayer(ReLULayer())
    #model_1.addlayer(LinearLayer(784, 784))
    #model_1.addlayer(ReLULayer())
    model_1.addlayer(LinearLayer(20, 10))
    model_2 = Sequential_2() 
    model_2.addlayer(FlattenLayer())
    model_2.addlayer(LinearLayer(784, 20))
    #model_2.addlayer(ReLULayer())
    #model_2.addlayer(LinearLayer(784, 784))
    #model_2.addlayer(ReLULayer())
    model_2.addlayer(LinearLayer(20, 10))
    model_3 = Sequential_3() 
    model_3.addlayer(FlattenLayer())
    model_3.addlayer(LinearLayer(784, 20))
    #model_3.addlayer(ReLULayer())
    #model_3.addlayer(LinearLayer(784, 784))
    #model_3.addlayer(ReLULayer())
    model_3.addlayer(LinearLayer(20, 10))
    model_4 = Sequential_4()  
    model_4.addlayer(FlattenLayer())
    model_4.addlayer(LinearLayer(784, 20))
    #model_4.addlayer(ReLULayer())
    #model_4.addlayer(LinearLayer(784, 784))
    #model_4.addlayer(ReLULayer())
    model_4.addlayer(LinearLayer(20, 10))


    classifier_1 = Classifier(model_1) #to make model work as a system by using Classifier()
    classifier_2 = Classifier(model_2)
    classifier_3 = Classifier(model_3)
    classifier_4 = Classifier(model_4)
        

        
        
    mnist = fetch_mldata('MNIST original')
    n = len(mnist.data)
    N = 10000
    indices = np.random.permutation(range(n))[:N]
    image = mnist.data[indices]
    image = image / 255.0 #normalization
    acc_array = np.empty(6)
    acc_array_1 = np.empty(6)
    acc_array_2 = np.empty(6)
    acc_array_3 = np.empty(6)
    acc_array_4 = np.empty(6)
    noise_rate_array = np.empty(6)
    image_tiny = np.zeros((image.shape[0],image.shape[1]/4))
    for d in range(0,26,5):
        random.seed(0)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if (random.randint(0,99) < d):
                    image[i][j] = random.random()
        '''for i in range(image_tiny.shape[0]):
            for j in range(image_tiny.shape[1]):
                image_tiny[i][j] = (image[i][j*4] + image[i][j*4+1] + image[i][j*4+2] + image[i][j*4+3])'''
      
        #image_tiny = np.reshape(image_tiny, (-1, 1, 14, 14))
        image_data = np.reshape(image, (-1, 1, 28, 28))
        label_original = mnist.target[indices]
        label = np.eye(10)[label_original.astype(int)]#one-hot expression
        label = np.argmax(label, axis=1)
        image_train, image_test, label_train, label_test = train_test_split(image_data,label,train_size=0.8)
        batchsize = 100
        ntrain = 8000
        ntest = 2000
        for e in range(epoch):
            print('epoch %d'%e)
            randinds = np.random.permutation(ntrain) #create an array which includes numbers 0 to ntrain randomly
            for i_train in range(0, ntrain, batchsize):
                ind = randinds[i_train:i_train+batchsize]
                x = image_train[ind]
                t = label_train[ind]
                start = time.time()
                loss_1, acc_1 = classifier_1.update(x, t) #make input pass through the classifier,doing BP, then update the parameters of each layer
                loss_2, acc_2 = classifier_2.update(x, t)
                loss_3, acc_3 = classifier_3.update(x, t)
                loss_4, acc_4 = classifier_4.update(x, t)
                end = time.time()
                print('train iteration %d, elapsed time %f, loss %f, acc %f'%(i_train//batchsize, end-start, loss_1, acc_1))
        start = time.time()
        acctest = 0
        acctest_1 = 0
        acctest_2 = 0
        acctest_3 = 0
        acctest_4 = 0
        losstest = 0
        for i_test in range(0, ntest, batchsize):
            x = image_test[i_test:i_test+batchsize]
            t = label_test[i_test:i_test+batchsize]
            y_1 = classifier_1.predict(x, t)
            y_2 = classifier_2.predict(x, t)
            y_3 = classifier_3.predict(x, t)
            y_4 = classifier_4.predict(x, t)
            y = (y_1 + y_2 + y_3 + y_4) / 4
            pred_1 = np.argmax(y_1, axis=1)
            pred_2 = np.argmax(y_2, axis=1)
            pred_3 = np.argmax(y_3, axis=1)
            pred_4 = np.argmax(y_4, axis=1)
            pred_1_re = np.reshape(pred_1, (pred_1.shape[0],1))
            pred_2_re = np.reshape(pred_2, (pred_2.shape[0],1))
            pred_3_re = np.reshape(pred_3, (pred_3.shape[0],1))
            pred_4_re = np.reshape(pred_4, (pred_4.shape[0],1))
            pred = np.concatenate([pred_1_re,pred_2_re], 1)
            pred = np.concatenate([pred,pred_3_re], 1)
            pred = np.concatenate([pred,pred_4_re], 1)
            pred = stats.mode(pred,axis=1)[0]
            pred = np.reshape(pred, (pred.shape[1],pred.shape[0]))
            acc = 1.0 * np.where(pred == t)[0].size / y.shape[0]
            acc_1 = 1.0 * np.where(pred_1 == t)[0].size / y_1.shape[0]
            acc_2 = 1.0 * np.where(pred_2 == t)[0].size / y_2.shape[0]
            acc_3 = 1.0 * np.where(pred_3 == t)[0].size / y_3.shape[0]
            acc_4 = 1.0 * np.where(pred_4 == t)[0].size / y_4.shape[0]
            loss = softmax_cross_entropy(y,t)
            acctest += int(acc * batchsize)
            acctest_1 += int(acc_1 * batchsize)
            acctest_2 += int(acc_2 * batchsize)
            acctest_3 += int(acc_3 * batchsize)
            acctest_4 += int(acc_4 * batchsize) 
            losstest += loss
        acctest /= (1.0 * ntest)
        acctest_1 /= (1.0 * ntest)
        acctest_2 /= (1.0 * ntest)
        acctest_3 /= (1.0 * ntest)
        acctest_4 /= (1.0 * ntest)
        losstest /= (ntest // batchsize)
        end = time.time()
        print('test, elapsed time %f, loss %f, acc %f'%(end-start, losstest, acctest_1))
        acc_array[d/5] = acctest
        acc_array_1[d/5] = acctest_1
        acc_array_2[d/5] = acctest_2
        acc_array_3[d/5] = acctest_3
        acc_array_4[d/5] = acctest_4
        noise_rate_array[d/5] = d
    print(acc_array)
    plt.bar(noise_rate_array,acc_array)
    plt.title("acc_comparison")
    plt.xlabel("Noise_rate")
    plt.ylabel("Accuracy")
    plt.savefig('acc_graph/acc_comparison.jpg')
        
if __name__ == '__main__': #not to execute main() when another python file imports this file
    epoch = 20 
    main(epoch)

        
