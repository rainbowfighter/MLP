#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
import numpy as np
import gc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def learn(network,samples, epochs=1000000, lrate=.0001, momentum=0.01):
        # Train 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            err = network.propagate_backward( samples['output'][n], lrate, momentum )      
            percent = (i/epochs)*100
            if (percent%1 == 0):
                print('learning: %.2d %% ' % (percent) + 'Error: %.5f' % err)
    
    # Example 4 : Learning sin(x) + cos(x2) function
    # -------------------------------------------------------------------------
    print ("Learning the sin(x1) + cos(x2) function")
    network = MLP(2,40,1)
    
    
    #---------------TRAINING SET---------------#
    pontok_szama = 200
    epochs = 1000
    training_set = np.zeros(pontok_szama*pontok_szama, dtype=[('input',  float, 2), 
                                                              ('output', float, 1)])
    tr_col = np.zeros(pontok_szama)
    
    tr_col = np.linspace(-2*np.pi, 2*np.pi, pontok_szama)
    
    tr_temp = []
    for i in range(tr_col.shape[0]):
        for j in range (tr_col.shape[0]):
            tr_temp.append([tr_col[i], tr_col[j]])
    training_set['input'] = np.array(tr_temp)
    training_set['output'] = (np.sin(training_set['input'][:,0]) + np.cos(training_set['input'][:,1]))/2.1
    mean = np.mean(training_set['input'])
    std = np.std(training_set['input'])
    training_set['input'] = (training_set['input'] - mean)/std
    
    #np.random.shuffle(training_set)
    #---------------/TRAINING SET---------------#
    
    #---------------LEARNING PHASE---------------#
    
    learn(network,training_set, epochs)
    
    #---------------/LEARNING PHASE---------------#
    
    #---------------TESTING PHASE---------------#
    ts_col = np.zeros(pontok_szama*pontok_szama)
    ts_col = np.linspace(-2*np.pi, 2*np.pi, pontok_szama)
    mean = np.mean(ts_col)
    std = np.std(ts_col)
    ts_col = (ts_col - mean)/std
    
    test_set = np.zeros(pontok_szama*pontok_szama, dtype=[('input',  float, 2), 
                                                         ('output', float, 1)])
                                                                                                       
    ts_temp = []
    for i in range(ts_col.shape[0]):
        for j in range (ts_col.shape[0]):
            ts_temp.append([ts_col[i], ts_col[j]])
    test_set['input'] = np.array(ts_temp)
    
    for i in range(test_set.shape[0]):
        test_set['output'][i] = network.propagate_forward(test_set['input'][i])
    graph = test_set['output'].reshape(pontok_szama,pontok_szama)
     
    #---------------/TESTING PHASE---------------#    
    
    #---------------DISPLAY PHASE---------------#    
    
    #Draw real function
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    X = tr_col
    Y = tr_col
    X, Y = np.meshgrid(X,Y)
    Z = training_set['output'].reshape(pontok_szama, pontok_szama)
    line = ax.plot_surface(X, Y, Z, color='blue')
    Z = graph
    line = ax.plot_surface(X, Y, Z, color='red')
    
    plt.show()
    
     #---------------/DISPLAY PHASE---------------#  
    
    gc.collect()