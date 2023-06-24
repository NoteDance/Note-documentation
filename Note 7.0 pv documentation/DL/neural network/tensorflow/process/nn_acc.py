import tensorflow as tf
import Note.nn.layer.dense as d
from Note.nn.process.optimizer import Momentum
from Note.nn.layer.flatten import flatten

# Define a neural network class
class nn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.optimizer=Momentum(0.07,0.7)
    
    
    def build(self):
        # Create two dense layers with relu and linear activations
        self.layer1=d.dense([784,128],activation='relu')
        self.layer2=d.dense([128,10])
        # Store the parameters of the layers in a list
        self.param=[self.layer1.param,self.layer2.param]
        return
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        data=flatten(data)
        output1=self.layer1.output(data)
        output2=self.layer2.output(output1)
        return output2
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)
    
    
    def accuracy(self,output,labels): #accuracy function,kernel uses it to calculate accuracy.
        return self.train_accuracy(labels,output)
    

    def opt(self,gradient):
        # Perform optimization on the parameters using the gradient
        param=self.optimizer.opt(gradient,self.param)
        return param