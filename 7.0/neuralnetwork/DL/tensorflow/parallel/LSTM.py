import tensorflow as tf
from Note.nn.layer.LSTM import LSTM
from Note.nn.layer.dense import dense
from Note.nn.parallel.optimizer import Adam
from Note.nn.Module import Module

# Define a recurrent neural network class with LSTM layers
class lstm:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=Adam()
        # Initialize a variable to keep track of the batch count
        self.bc=tf.Variable(0, dtype=tf.float32)
    

    def build(self):
        # Create two LSTM layers with 50 hidden units each
        self.lstm1=LSTM(50,3,return_sequence=True) # First LSTM layer that returns the full sequence
        self.lstm2=LSTM(50,50) # Second LSTM layer that returns the last output only
        # Create a dense layer with 10 output units
        self.dense=dense(10,50)
        # Store the parameters of the layers in a list
        self.param=Module.param
        return
    

    def fp(self,data):
        # Perform forward propagation on the input data
        x=self.lstm1.output(data) # First LSTM layer output
        x=self.lstm2.output(x) # Second LSTM layer output
        output=self.dense.output(x) # Dense layer output
        return output
    
    
    def loss(self, output, labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels, output)
    
    
    def opt(self,gradient):
        # Perform optimization on the parameters using the gradient and the batch count
        param=self.optimizer.opt(gradient,self.param,self.bc[0])
        return param                       
