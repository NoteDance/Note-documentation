import tensorflow as tf # import TensorFlow library
import Note.nn.layer.dense as d # import Note's dense layer module
from Note.nn.layer.flatten import flatten # import Note's flatten layer function
from Note.nn.process.optimizer import Momentum # import Note's momentum optimizer module
from Note.nn.process.assign_device import assign_device # import the function to assign device according to the process index and the device type


class nn:               # A neural network class example, allocate device for multiple threads
    def __init__(self): # initialize the network
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss object, kernel uses it to calculate loss. Here we use sparse categorical crossentropy with logits as output
        self.optimizer=Momentum(0.07,0.7) # optimizer, kernel uses it to optimize. Here we use a custom momentum optimizer with learning rate 0.07 and momentum 0.7
        self.info='example' # some information about the network
    
    
    def build(self): # build function, kernel uses it to create the network layers
        # Create two dense layers with relu and linear activations
        self.layer1=d.dense([784,128],activation='relu') # the first layer with 784 input units and 128 output units and ReLU activation
        self.layer2=d.dense([128,10]) # the second layer with 128 input units and 10 output units and linear activation
        # Store the parameters of the layers in a list
        self.param=[self.layer1.param,self.layer2.param] # parameter list of both layers, kernel uses it list for backpropagation 
        return
    
    
    def fp(self,data,p): # forward propagation function, kernel uses it for forward propagation
        with tf.device(assign_device(p,'GPU')): # assign the device according to the process index p
            data=flatten(data) # flatten the data to a vector of 784 elements
            output1=self.layer1.output(data) # pass the data through the first layer and get the output
            output2=self.layer2.output(output1) # pass the output of the first layer through the second layer and get the final output logits
        return output2
    
    
    def loss(self,output,labels,p): # loss function, kernel uses it to calculate loss
        with tf.device(assign_device(p,'GPU')): # assign the device according to the process index p
            return self.loss_object(labels,output) # return the mean softmax cross entropy loss between labels and output
    
    
    def opt(self,gradient): # optimization function, kernel uses it to optimize parameter
        # Perform optimization on the parameters using the gradient
        param=self.optimizer.opt(gradient,self.param) # apply the Note's momentum optimizer to update the parameters using the gradient
        return param
