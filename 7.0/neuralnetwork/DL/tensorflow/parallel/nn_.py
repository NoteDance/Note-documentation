import tensorflow as tf
from Note.nn.parallel.optimizer import Momentum
from Note.nn.layer.flatten import flatten

# Define a neural network class
class nn:
    def __init__(self):
        # Initialize the weights and biases of three layers with random values
        self.weight1=tf.Variable(tf.random.normal([784,64],dtype='float64'))
        self.bias1=tf.Variable(tf.random.normal([64],dtype='float64'))
        self.weight2=tf.Variable(tf.random.normal([64,64],dtype='float64'))
        self.bias2=tf.Variable(tf.random.normal([64],dtype='float64'))
        self.weight3=tf.Variable(tf.random.normal([64,10],dtype='float64'))
        self.bias3=tf.Variable(tf.random.normal([10],dtype='float64'))
        # Store the parameters of the layers in a list
        self.param=[self.weight1,self.weight2,self.weight3,self.bias1,self.bias2,self.bias3]
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=Momentum(0.07,0.7)
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        data=flatten(data)
        layer1=tf.nn.relu(tf.matmul(data,self.weight1)+self.bias1) # First layer with relu activation
        layer2=tf.nn.relu(tf.matmul(layer1,self.weight2)+self.bias2) # Second layer with relu activation
        output=tf.matmul(layer2,self.weight3)+self.bias3 # Output layer with linear activation
        return output
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)
    

    def opt(self,gradient):
        # Perform optimization on the parameters using the gradient
        param=self.optimizer.opt(gradient,self.param)
        return param