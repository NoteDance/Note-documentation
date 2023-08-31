import tensorflow as tf
from Note.nn.layer.LSTM import LSTM
from Note.nn.layer.self_attention import self_attention
from Note.nn.layer.dense import dense
from Note.nn.layer.emdedding import embedding

# Define a LSTM self attention neural network class
class self_LSTM:
    def __init__(self,vocab_size,embed_size,num_heads,num_layers):
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.num_heads=num_heads
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    
    
    def build(self):
        # Create an embedding layer to map input tokens to vectors
        self.embedding_layer = embedding(self.embed_size,self.vocab_size)
        # Create a LSTM layer to process the input sequence
        self.lstm_layer = LSTM(self.embed_size,self.embed_size,return_sequence=True)
        # Create a self attention layer to attend to the LSTM output
        self.self_attention_layer = self_attention(self.embed_size,self.embed_size,num_heads=self.num_heads)
        # Create a linear layer to map the attention output to logits
        self.logits_layer = dense(self.vocab_size，self.embed_size)
        # Store the parameters of the layers in a list
        self.param = [self.embedding_layer.param]
        self.param.extend(self.lstm_layer.param)
        self.param.extend(self.self_attention_layer.param)
        self.param.extend(self.logits_layer.param)
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        x = self.embedding_layer.output(data) # shape: (batch_size, seq_len, embed_size)
        x = self.lstm_layer.output(x) # shape: (batch_size, seq_len, embed_size)
        x, attention_weights = self.self_attention_layer.output(x,self.num_heads) # shape: (batch_size, seq_len, embed_size)
        output = self.logits_layer.output(x) # shape: (batch_size, seq_len, vocab_size)
        return output
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)
