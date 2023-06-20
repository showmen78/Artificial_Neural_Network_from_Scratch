from utility import *
import numpy as np

'''
    Layer class:
        Does the following  calculations for a single layer
        
        METHODS:
            1-> Forward Propagation
            2-> Back Propagation
            3-> Calculate Cost
            4-> Update Parameters
'''

class Layer:
    def __init__(self,w_size,last_layer,activation):
        '''
            input:
                w_size: a list [no of input node, no of output node]
                last_layer: is this layer the last layer(True/False)
                activation: string (activation used in this layer)
                
        '''
        
        #parameters of this layer
        self.w= np.random.randn(w_size[1],w_size[0])
        self.dw=[]
        
        self.b= np.zeros((w_size[1],1))
        self.db=[]
        
        self.z=[]
        self.dz=[]
        
        self.last_layer= last_layer # is this the last layer
        
        self.x=[]  #input of this layer
        self.A= [] #output of this layer
        
        #activation used in this layer
        self.activation=activation




    def forward_propagation(self,A):
        '''
            input:
                A: input of this layer


            output:
                self.A: The output of this layer( activation(self.z))
        '''
        self.x=A
        self.z= np.array(self.w.dot(A)+ self.b)
        self.A= np.array(eval(self.activation+'(self.z)'))

        return self.A


    def back_propagation(self,y,w_next,dz_next,batch_size):
        '''
            input:
                
                y:an array of actual output of shape (no of class,no of samples)
                w_next: w parameter of the next layer(is not used in the last layer)
                dz_next: dz parameter of the next layer(is not used in the last layer)
                batch_size : batch size

            output:
                dw,dz : an array , value of dw and dz of this layer
        '''
        # for the last layer
        if self.last_layer:
            self.dz= np.array(self.A-y)
            #in the last layer, no of samples may not be the same as batch size
            batch_size= y.shape[1]
            
        # if not last layer
        else:
            #self.dz= self.np.array(w_next.T.dot(dz_next)) * derivative_of_sigmoid(self.z)
            self.dz= np.array(w_next.T.dot(dz_next)) * eval('derivative_of_'+self.activation+'(self.z)')


        #self.dw= (1/batch_size)*np.array(self.dz.dot(self.input.T))
        self.dw= (1/batch_size)*np.array(self.dz.dot(self.x.T))
        self.db= (1/batch_size)*np.sum(self.dz,axis=1,keepdims=True)

        return np.array(self.w),np.array(self.dz)

    def calculate_cost(self,y,_type):
        '''
            input:
                batch_size: batch size
                y: actual value of the output
                _type: type of cost function (binary classification / multiclassification)

            output: return the cost value
            Note: only the last layer calls this method
        '''
        #for binary classification
        if _type=='binary_classification':
            return np.sum((y*np.log(self.A))+(1-y)*np.log(1-self.A))*(-1/y.shape[1]) 

        #for multiclassification problem
        if _type=='multi_classification':
            return np.sum(y*np.log(self.A))/(-1/y.shape[1]) 

    def update_parameters(self,lr):
        '''
            lr: learning rate
            Note: Updates the parameter 
        '''
        self.w = self.w- (lr*self.dw)
        self.b= self.b- (lr*self.db)