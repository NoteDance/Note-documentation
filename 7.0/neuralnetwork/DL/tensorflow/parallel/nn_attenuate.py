import tensorflow as tf
from tensorflow.python.util import nest
import Note.nn.layer.dense as d
from Note.nn.parallel.optimizer import Momentum
from Note.nn.layer.flatten import flatten
from Note.nn.accuracy import sparse_categorical_accuracy
from Note.nn.Module import Module

# Define a neural network class with gradient attenuation
class nn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=Momentum(0.07,0.7)
        # Initialize a variable to keep track of the number of optimization steps
        self.opt_counter=tf.Variable(tf.zeros(3,dtype=tf.float32))
    
    
    def build(self):
        # Create two dense layers with relu and linear activations
        self.layer1=d.dense(128,784,activation='relu')
        self.layer2=d.dense(10,128)
        # Store the parameters of the layers in a list
        self.param=Module.param
        return
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        data=flatten(data) # Flatten the data to a one-dimensional vector
        output1=self.layer1.output(data) # Apply the first layer to the data and get the output
        output2=self.layer2.output(output1) # Apply the second layer to the output of the first layer and get the final output
        return output2
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output) # Use the loss function to calculate the loss
    
    
    def accuracy(self,output,labels): #accuracy function,kernel uses it to calculate accuracy.
        return sparse_categorical_accuracy(labels,output)
    
    
    def attenuate(self,gradient,oc,p):  #gradient attenuation function,kernel uses it to calculate attenuation coefficient.
        # Apply an exponential decay to the gradient based on the optimization counter
        ac=0.9**oc[0][p]                   #ac:attenuation coefficient
        gradient_flat=nest.flatten(gradient) # Flatten the gradient to a one-dimensional vector
        for i in range(len(gradient_flat)):  #oc:optimization counter
            gradient_flat[i]=ac*gradient_flat[i]  #p:process number
        gradient=nest.pack_sequence_as(gradient,gradient_flat) # Restore the gradient to its original shape
        return gradient  
    

    def opt(self,gradient):
        # Perform optimization on the parameters using the gradient
        param=self.optimizer.opt(gradient,self.param) # Use the optimizer to update the parameters
        return param                            
