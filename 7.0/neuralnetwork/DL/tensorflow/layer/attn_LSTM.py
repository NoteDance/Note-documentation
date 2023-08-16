import tensorflow as tf
from Note.nn.layer.LSTM import LSTM
from Note.nn.layer.dense import dense
from Note.nn.layer.attention import attention

# Define a recurrent neural network class with LSTM layers and attention
class attn_lstm:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    

    def build(self):
        # Create two LSTM layers with 50 hidden units each
        self.lstm1=LSTM(weight_shape=(3,50),return_sequence=True) # First LSTM layer that returns the full sequence
        self.lstm2=LSTM(weight_shape=(50,50),) # Second LSTM layer that returns the last output only
        # Create a dense layer with 10 output units and softmax activation
        self.dense=dense([100, 10],activation='softmax')
        # Create an attention object with weight shape (50, 50)
        self.attn=attention((50,50))
        # Store the parameters of the layers and the attention in a list
        self.param=[self.lstm1.param,
                      self.lstm2.param,
                      self.dense.param,
                      self.attn.param] # Add the attention parameters to the list
        return
    

    def fp(self,data):
        # Perform forward propagation on the input data
        x=self.lstm1.output(data) # First LSTM layer output, shape (B, N, 50)
        y=self.lstm2.output(x) # Second LSTM layer output, shape (B, 50)
        # Use the attention object to compute the context vector from x and y, shape (B, 50)
        context_vector,_,_=self.attn.output(x,y)
        # Concatenate the context vector and the second LSTM output, shape (B, 100)
        z=tf.concat([context_vector,y],axis=-1)
        output=self.dense.output(z) # Dense layer output with softmax activation, shape (B, 10)
        return output
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)