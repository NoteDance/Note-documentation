import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.layer.flatten import flatten
from Note.nn.Module import Module

"""
This is an example of using the Note layer module.
"""

# Define a neural network class
class nn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam() #optimizer,kernel uses it to optimize.
    
    
    def build(self):
        # Create two dense layers with relu and linear activations
        self.layer1=dense(128,activation='relu')
        self.layer2=dense(10)
        self.flatten=flatten()
        # Store the parameters of the layers in a list
        self.param=Module.param
        return
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        data=self.flatten.output(data)
        output1=self.layer1.output(data)
        output2=self.layer2.output(output1)
        return output2
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)