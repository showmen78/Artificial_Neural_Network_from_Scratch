from layer import Layer
import numpy as np
import math


'''
    The Artificial Neural Network Model

    METHODS:
        TRAIN: Trains the model 
        PREDICT: Give the prediction of train and test data 
'''

class Ann:
    def __init__(self,neurons,activation,_type='binary_classification'):
        '''
            input:
                neurons: a list of integer , contains the number of neurons in each layer eg(10,12,10)
                activation: a list containing the name of activations for each layer
                _type: a string, binary classification or multi classification
        '''
        
        self.neurons= neurons
        self.activation= activation
       
        #contains all the layer objects
        self.layers=[]
        self._type= _type
        
        
        
    def train(self,_input,y_acc,batch_size,test_input=np.zeros((1,4)),test_label=np.zeros((1,4)),lr=0.1,iterations=1,show_acc=[True,True]):
            '''
                input:
                    _input: a numpy array, contains the input features , shape(input_features,sample no)
                    y_acc : an array of actual output, shape(no of class, sample no)
                    test_input: an array , contains the test input features , shape(input_features,test sample no)
                    test_label: an array , actual output of the test cases , shape(no of class,no of test sample)
                    
                    batch_size: an int, batch size for training
                    lr : a float , learning rate
                    iterations: an integer value, training iterations
                    show_acc: a list of bolean value ,show accuracy of [train,test] after each iteration
            '''

            #if there is no layer then create layer objects at first

            if len(self.layers)==0:
                #first value of the self.neuron will be the row of input array.
                self.neurons.insert(0,_input.shape[0])
                
                #creating layer objects
                for n in range(1,len(self.neurons)):
                    self.layers.append(Layer([self.neurons[n-1],self.neurons[n]],n== len(self.neurons)-1,self.activation[n-1]))

            #stores the shape of the original input
            _input_shape= [_input.shape[0],_input.shape[1]]
            
            #converting the _input and outputs to mini batches
            _input = [_input[:,j*batch_size:(j*batch_size)+batch_size] for j in range(math.ceil(_input.shape[1]/batch_size))]      
            y_acc= [y_acc[:,j*batch_size:(j*batch_size)+batch_size] for j in range(math.ceil(y_acc.shape[1]/batch_size))]
            
            
            #start the training 
            for i in range(iterations):
                cost =0 #contains the cost value of a whole iteration
                #for each batch
                for batch_x,batch_y in zip(_input,y_acc):
                    #input of the first layer
                    A = batch_x

                    #forward propagation for all layers
                    for layer in self.layers:
                        A= layer.forward_propagation(A)
                        
                    #calculate and add cost
                    cost += self.layers[-1].calculate_cost(batch_y,self._type)

                    #back propagation for all layers
                    w=dz=0
                    for layer in self.layers[::-1]:
                        w,dz= layer.back_propagation(batch_y,w,dz,batch_size)
                        #update parameters
                        layer.update_parameters(lr)
                
                # after each iteration
                #print train accuracy
                if show_acc[0]:
                    print('iter:{} - train_cost:{} - train_acc:{} -'.format(i,round(cost,2),self.predict(_input,y_acc,batch_size,True)[0]),
                        end=' ')
                
                #print test accuracy
                if show_acc[1]:
                    print('test_acc:{}'.format(self.predict(test_input,test_label,batch_size,False)[0]),end='')
                print() #draw a new line



    def predict(self,x,y,batch_size,train=False):

        '''
            input:
                x: an array of input features
                y_acc: an array of actual outputs
                batch_size: batch size
                train: is this training data or testing data (True for training data)

            output:
                return a list , [accuracy,predicted output]
        '''
        
        
        #divide only the test data into mini batches, train data are divided into batches in the train function
        if train==False:
            x = [x[:,j*batch_size:(j*batch_size)+batch_size] for j in range(math.ceil(x.shape[1]/batch_size))]
            y= [y[:,j*batch_size:(j*batch_size)+batch_size] for j in range(math.ceil(y.shape[1]/batch_size))]
        #contains the accuracy
        acc=[]
        #contains the predicted output
        y_pred=[]
        predicted_output= np.zeros((1,1))
        #counts how many samples are there
        m=0 
        #run forward propagation
        for batch_x ,batch_y in zip(x,y):
            a= batch_x
            for l in range(len(self.layers)):
                a= self.layers[l].forward_propagation(a)
            
            if np.sum(predicted_output)==0:
                predicted_output=a
                
            else:
                predicted_output= np.column_stack((predicted_output, a))
            
            if self._type== 'multi_classification':
                a= np.argmax(a,0)
                y_pred= np.argmax(batch_y,0)
                m+= a.shape[0]
                
            if self._type == "binary_classification":
                a=np.array(a>.5,dtype='float')
                y_pred= batch_y
                m+= a.shape[1]
                

            #m += a.shape[0]  #add the sample number in each batch to calculate avg accuracy
            acc.append(np.sum(np.array(a==y_pred,dtype='float')))

    
        return [round(100*sum(acc)/m,2),predicted_output]
       
       