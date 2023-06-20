#utility function

'''
    contains the following activation functions and their derivatives

    1-> RELU
    2-> SIGMOID
    3-> TANH
    4-> SOFTMAX


'''

import numpy as np

def relu(z):
    return np.maximum(z,0)

def derivative_of_relu(z):
    return np.array(z>0,dtype='float')

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative_of_sigmoid(z):
    sg= sigmoid(z)
    return sg*(1-sg)

def tanh(z):
    return np.tanh(z)

def derivative_of_tanh(z):
    return 1-np.power(tanh(z),2)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=0)