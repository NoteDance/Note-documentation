import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.flatten import flatten
from Note.nn.Model import Model

"""
This is an example of using the Note layer module.
"""

# Define a convolutional neural network class
class cnn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    
    
    def build(self):
        # Create three convolutional layers with relu activations
        self.conv1=conv2d(32,[3,3],strides=(1,1),padding='SAME',activation='relu')
        self.conv2=conv2d(64,[3,3],strides=(2,2),padding='SAME',activation='relu')
        self.conv3=conv2d(64,[3,3],strides=(1,1),padding='SAME',activation='relu')
        self.flatten=flatten()
        # Create two dense layers with relu and linear activations
        self.dense1=dense(64,activation='relu')
        self.dense2=dense(10)
        # Store the parameters of the layers in a list
        self.param=Model.param
        return
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        x=self.conv1(data) # First convolutional layer
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID') # First max pooling layer
        x=self.conv2(x) # Second convolutional layer
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID') # Second max pooling layer
        x=self.conv3(x) # Third convolutional layer
        x=self.flatten(x) # Flatten the output to a vector
        x=self.dense1(x) # First dense layer with relu activation
        output=self.dense2.output(x) # Output layer with linear activation
        return output
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)