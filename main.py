import numpy as np
from Ann import Ann


'''
    The main function does the following 

    1-> Import the data
    2-> Divide training and test data
    3-> Normalize the data
    4-> Train the Ann model
    5-> Predict using the Ann model
'''


#import the train and test data
X_train = np.loadtxt('./data/train_X.csv', delimiter = ',').T
Y_train = np.loadtxt('./data/train_label.csv', delimiter = ',').T

X_test = np.loadtxt('./data/test_X.csv', delimiter = ',').T
Y_test = np.loadtxt('./data/test_label.csv', delimiter = ',').T

#### show a random image #####

# index = random.randrange(0, X_train.shape[1])
# plt.imshow(X_train[:, index].reshape(28, 28), cmap = 'gray')
# plt.show()



## divide the data into train and test set and shuffle them
data={}
for i in range(X_train.shape[1]):
    data[str(i)]= {'x':X_train[:,i],'y':Y_train[:,i]}
    
#randomizing the training data 
keys= list(data.keys())
np.random.shuffle(keys)
X=[]
Y=[]
for i in keys:
    X.append(data[i]['x'])
    Y.append(data[i]['y'])


# normalizing the data
    
X= np.array(X).T/255 #final train data #dividing by 255 to normalize the value between 0 and 1
Y= np.array(Y).T #final test data   

X_test= np.array(X_test)/255   #dividing by 255 to normalize the value between 0 and 1
Y_test= np.array(Y_test)


#iterations no
iterations=10
#batch size for training
batch_size=10

#learing rate for training
learning_rate=.25

# #calling the ANN model and TRAINING it
a= Ann([10],['softmax'],'multi_classification')
a.train(X,Y,batch_size,X_test,Y_test,lr=learning_rate, iterations=iterations,show_acc=[True,True])


###  PREDICTION  ###
prediction=a.predict(X_test,Y_test,20,False)

# accuracy of the test data
#print(prediction[0])

#actual prediction of the test data
#print(prediction[1])
