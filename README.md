# Artificial Neural Network from Scratch

This repository contains a Python project named "Artificial Neural Network from Scratch" which implements a simple artificial neural network to classify digits ```( 0-9 ) ``` from **```Scratch```**

## Project Structure

The project consists of the following files:

1. **Layer.py**: This file contains the implementation of a single layer in the neural network. It includes methods for forward propagation, backpropagation, cost calculation, and parameter updates.

``` 
forward_propagation(input_data): Run forward propagation and return output after activation ...
z= wx+b
returns A= activation(z)

 ```

 ```  
 back_propagation(actual_output,w_of_the_next_layer,dz_of_the_next_layer,batch_size): Implements the backpropagation of a single layers.
 -- calculates dz,dw,db,dx 
 -- updates parameter (w and b)
 ```

 ``` 
 calculates_cost(actual_output,_type): calculates the cost 

 _type: binary_classification, multi_classification .

COST FUNCTION:
binary_classification: -(y*log(A)+(1-y)*log(1-A))/batch_size
multi_classification: summation(y*log(A))

 ```



2. **Ann.py**: This file represents the Artificial Neural Network (ANN) model. It provides two main methods, `train` and `predict`, for training the model on the MNIST dataset and making predictions,respectively.

``` 
** train(train_input,train_label,batch_size,test_data,test_label,learning_rate,iterations,show_acc):
-- Trains the Ann model 
-- show_acc [bool,bool]: If true , it will show the training and test accuracy after each iterations.
First value is for train data and second value is for test data.

** predict(x,y,batch_size,train=bool): After training , predicts based on x and y data.

-- x: Input for predictions
-- y: Actual label
-- batch_size : batch_size
-- train (bool): False if x and y are test data, True otherwise
-- returns : [accuracy,actual_prediction]
 ```




3. **utility.py**: This file contains utility functions, including various activation functions and their derivatives, which are used in the neural network.
```
relu(x) , sigmoid(x) , tanh(x) , softmax(x) , derivatives_of_relu(x),  derivatives_of_sigmoid(x), derivatives_of_tanh(x) 

 ```

4. **main.py**: This file serves as the entry point of the project. It imports the MNIST dataset, preprocesses the data, and calls the ANN model to train it using the preprocessed data and  make predictions using `test data`

## Dataset

The  dataset that is  used for digit classification in this project  consists of a large set of 28x28 pixel grayscale images of handwritten digits from 0 to 9. 
In `train_X.csv (train_data)` there are 1000 samples .
In `test_X.csv (test_data)` there are 350 samples.

## Getting Started

To run this project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/showmen78/Artificial-Neural-Network-from-Scratch.git
```

2. Install the required dependencies. You can use `pip` to install them:

```
pip install numpy
pip install matplotlib
```

3. Run the `main.py` file:

```
python main.py
```

## Usage

To use the Artificial Neural Network model in your own projects, you can follow these steps:

1. Import the `Ann` class from `Ann.py`:

```python
from Ann import Ann
```

2. Create an instance of the `Ann` class:

```python 
ann = Ann(neurons,activation,_type='binary_classification')

```

```
-- neurons([int]): neurons in each layer. eg [10,10]
-- activation([str]): list of name of the activation function for each layer. eg ["relu","softmax"]
-- _type(str): 'binary_classification'/'multi_classification'
```

3. Train the model on your own dataset:

```python
ann.train(X_train, y_train,batch_size,X_test,Y_test, learning_rate,iterations,show_acc)
```

4. Make predictions using the trained model:

```python
predictions = ann.predict(X_test,Y_test,batch_size,train=False)
```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Special thanks to the creators of the  dataset for providing the data used in this project.

