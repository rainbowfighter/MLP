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
    import matplotlib.pyplot as plt

    def learn(network,samples, epochs=1000000, lrate=.1, momentum=0.1):
        # Train 
        for i in range(epochs):
            percent = (i/epochs)*100
            if (percent%1 == 0):
                print('learning: %d %%' % (percent))
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )      
        
    
    # Example 4 : Learning sin(x)
    # -------------------------------------------------------------------------
    print ("Learning the sin function")
    network = MLP(2,20,20,1)
    
    
    #---------------TRAINING SET---------------#
    pontok_szama = 1000
    epochs = 1000000   
    training_set = np.zeros(pontok_szama, dtype=[('input',  float, 2), ('output', float, 1)])
    tr_col1 = np.zeros(pontok_szama)
    tr_col2 = np.zeros(pontok_szama)
    tr_col3 = np.zeros(pontok_szama)
    
    tr_col1 = np.linspace(-2*np.pi, 2*np.pi, pontok_szama)
    tr_col3 = np.sin(tr_col1)
    tr_col2[0] = 0
    i = 1
    while (i < tr_col1.shape[0]):
        tr_col2[i] = tr_col3[i-1]
        i += 1
    
    for i in range(training_set.shape[0]):
        training_set['input'][i] = (tr_col1[i],tr_col2[i])
        training_set['output'][i] = tr_col3[i]
    #---------------/TRAINING SET---------------#
    
    #---------------LEARNING PHASE---------------#
    
    learn(network,training_set, epochs)
    
    #---------------/LEARNING PHASE---------------#
    
    #---------------TESTING PHASE---------------#
    test_set = np.zeros(pontok_szama, dtype=[('input',  float, 2), ('output', float, 1)])
    ts_col1 = np.zeros(pontok_szama)
    
    ts_col1 = np.linspace(-2*np.pi, 2*np.pi, pontok_szama)
    test_set['input'][0] = (ts_col1[0], 0)
    test_set['output'][0] = network.propagate_forward(test_set['input'][0])
    
    for i in range(1,ts_col1.shape[0]):
       test_set['input'][i] = (ts_col1[i], test_set['output'][i-1])
       test_set['output'][i] = network.propagate_forward(test_set['input'][i]) 
     
    #---------------/TESTING PHASE---------------#    
    
    #---------------DISPLAY PHASE---------------#    
    
    #Draw real function
    plt.figure(figsize=(10,5))
    x,y = tr_col1,tr_col3
    plt.plot(x,y,'.',color='b',lw=2)
    x,y = ts_col1, test_set['output']
    plt.plot(x,y,'.',color='g',lw=2)
    plt.show()
    
     #---------------/DISPLAY PHASE---------------#  
    
    gc.collect()