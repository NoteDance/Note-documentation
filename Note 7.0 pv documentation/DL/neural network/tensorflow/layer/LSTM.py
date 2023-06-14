import tensorflow as tf
from Note.nn.layer.LSTM import LSTM
from Note.nn.layer.dense import dense

# Define a recurrent neural network class with LSTM layers
class lstm:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    

    def build(self):
        # Create two LSTM layers with 50 hidden units each
        self.lstm1=LSTM(weight_shape=(3,50),return_sequence=True) # First LSTM layer that returns the full sequence
        self.lstm2=LSTM(weight_shape=(50,50),) # Second LSTM layer that returns the last output only
        # Create a dense layer with 10 output units
        self.dense=dense([50, 10])
        # Store the parameters of the layers in a list
        self.param=[self.lstm1.param,
                      self.lstm2.param,
                      self.dense.param,
                      ]
        return
    

    def fp(self,data):
        # Perform forward propagation on the input data
        x=self.lstm1.output(data) # First LSTM layer output
        x=self.lstm2.output(x) # Second LSTM layer output
        output=tf.matmul(x,self.dense.weight)+self.dense.bias # Dense layer output
        return output
    
    
    def loss(self, output, labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels, output)
