#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
import struct as st
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[371]:


class NN:
    
    def __init__(self):
        self.st = [784, 128, 10]
        self.W = {}
        self.b = {}
        self.dW = {}
        self.db = {}
        self.h = {}
        self.z = {}
        self.delta = {}
        self.iter = 0
        self.alpha = 0.25
        self.samples = 60000
        
    def f(self, x):
        return 1.0 / (1 + 1.0 * np.exp(-x))
    
    def f_d(self, x):
        return np.multiply(self.f(x), (1 - self.f(x)))
    
    def init_weights(self):        
        for l in range(1, len(self.st)):
            self.W[l] = np.random.uniform(-2,2,(self.st[l], self.st[l - 1]))
            self.b[l] = np.random.uniform(-2,2,(self.st[l], 1))
    
    def init_dels(self):    
        for l in range(1, len(self.st)):
            self.dW[l] = np.zeros((self.st[l], self.st[l - 1]))
            self.db[l] = np.zeros((self.st[l], 1))
    
    def feed_forward(self, inp):
        self.h[1] = inp
        for l in range(1, len(self.st)):
            self.z[l + 1] = self.W[l] @ self.h[l] + self.b[l]
            self.h[l + 1] = self.f(self.z[l + 1])
    
    def calc_d(self, out):
        self.delta[len(self.st)] = (self.h[len(self.st)] - out) 
        for l in range(len(self.st) - 1, 1, -1):
            self.delta[l] = (np.transpose(self.W[l])).dot(self.delta[l + 1])
            self.delta[l] = self.delta[l] * self.f_d(self.z[l])
    
    def accuracy(self, data):
        self.samples = np.shape(data)[0]
        ac = 0
        for i in range(self.samples):
            inp_image = np.reshape(data[i,1:].T, (784,1))
            inp_image = inp_image / 255.0
            self.feed_forward(inp_image)
            ind = int(data[i,0])
            ch = np.argmax(self.h[len(self.st)])
            if ind == ch:
                ac = ac + 1
        ac *= 100.0
        ac /= self.samples
        return ac
    
    def train(self, data):
        self.iter = 100
        self.alpha = 1
        self.samples = 10000
        acc = 0
        self.init_weights()
        for cnt in range(self.iter):
            self.init_dels()
            acc = acc * 100.0
            acc = acc / 60000
            print("Epoch: " + str(cnt) + " Running Accuracy: " + str(acc)) 
            acc = 0
            for i in range(60000):
                inp_image = np.reshape(data[i,1:].T, (784,1))
                inp_image = inp_image / 255.0
                self.feed_forward(inp_image)
                out = np.zeros((10,1))
                ind = int(data[i,0])
                out[ind] = 1
                ch = np.argmax(self.h[len(self.st)])
                self.calc_d(out)
                if ind == ch:
                    acc = acc + 1
                for l in range(1, len(self.st)):
                    self.db[l] = self.db[l] + self.delta[l + 1]
                    self.dW[l] = self.dW[l] +  self.delta[l + 1].dot(np.transpose(self.h[l]))
                if (i + 1) % 10000 == 0:
                    val = -(self.alpha * 1.0) / self.samples
                    for l in range(1, len(self.st)):
                        self.W[l] = self.W[l] + val * self.dW[l]
                        self.b[l] = self.b[l] + val * self.db[l]
                    self.init_dels()


# In[362]:


data = np.loadtxt("mnist_train.csv" , delimiter = ",")
data2 = np.loadtxt("mnist_test.csv" , delimiter = ",")


# In[372]:


new = NN()
new.train(data)
print("Accuracy on training data:")
print(new.accuracy(data))
print("Accuracy on testing data:")
print(new.accuracy(data2))


# In[ ]:




