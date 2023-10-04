import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.flatten import flatten
from Note.nn.parallel.optimizer import Adam
from Note.nn.layer.LMix import lmix
from Note.nn.Module import Module

# Define a convolutional neural network class with mixup augmentation
class cnn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Initialize the alpha parameter for mixup
        self.alpha=1.0
    
    
    def build(self):
        # Create three convolutional layers with relu activations
        self.conv1=conv2d(32,[3,3],3,strides=(1,1),padding='SAME',activation='relu')
        self.conv2=conv2d(64,[3,3],32,strides=(2,2),padding='SAME',activation='relu')
        self.conv3=conv2d(64,[3,3],64,strides=(1,1),padding='SAME',activation='relu')
        # Create two dense layers with relu and linear activations
        self.dense1=dense(64,64*4*4,activation='relu')
        self.dense2=dense(10,64)
        self.optimizer=Adam()
        # Store the parameters of the layers in a list
        self.param=Module.param
        return
    
    
    def data_func(self,data_batch,labels_batch):
        # Apply mixup augmentation on the input data and labels
        return lmix(data_batch,labels_batch,self.alpha)
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        x=self.conv1.output(data) # First convolutional layer
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID') # First max pooling layer
        x=self.conv2.output(x) # Second convolutional layer
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID') # Second max pooling layer
        x=self.conv3.output(x) # Third convolutional layer
        x=flatten(x) # Flatten the output to a vector
        x=tf.nn.relu(tf.matmul(x,self.dense1.weight)+self.dense1.bias) # First dense layer with relu activation
        x=tf.nn.dropout(x,rate=0.5) # Apply dropout to prevent overfitting
        output=tf.matmul(x,self.dense2.weight)+self.dense2.bias # Output layer with linear activation
        output=tf.nn.dropout(output,rate=0.5) # Apply dropout to prevent overfitting
        return output
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)
    
    
    def opt(self,gradient):
        # Perform optimization on the parameters using the gradient and the batch count
        param=self.optimizer.opt(gradient,self.param,self.bc[0])
        return param                         
