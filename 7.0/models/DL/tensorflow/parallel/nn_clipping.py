import tensorflow as tf
from Note.nn.layer.dense import dense
from Note.nn.parallel.optimizer import SGD
from Note.nn.layer.flatten import flatten
from Note.nn.gradient_clipping import gradient_clipping
from Note.nn.Model import Model

# Define a neural network class
class nn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    
    def build(self):
        # Create two dense layers with relu and linear activations
        self.layer1=dense(128,784,activation='relu')
        self.layer2=dense(10,128)
        self.flatten=flatten()
        self.optimizer=SGD()
        # Store the parameters of the layers in a list
        self.param=Model.param
        return
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        data=self.flatten(data)
        output1=self.layer1(data)
        output2=self.layer2(output1)
        return output2
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)
    
    
    def gradient(self,tape,loss):
        # Compute the gradients of the loss with respect to the parameters
        grads=tape.gradient(loss,self.param)
        # Clip the gradients by value to avoid exploding gradients
        return gradient_clipping(grads,'value',1.0)
    

    def opt(self,gradient):
        # Perform optimization on the parameters using the gradient
        param=self.optimizer(gradient,self.param)
        return param