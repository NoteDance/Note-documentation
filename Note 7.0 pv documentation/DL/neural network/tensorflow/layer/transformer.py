import tensorflow as tf
from Note.nn.layer.transformer import transformer
from Note.nn.layer.dense import dense
from Note.nn.layer.emdedding import embedding
from Note.nn.positional_encoding import positional_encoding

# Define a positional encoding transformer neural network class
class Transformer:
    def __init__(self,vocab_size,embed_size,num_heads,num_layers,max_len):
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.max_len=max_len
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    
    
    def build(self):
        # Create an embedding layer to map input tokens to vectors
        self.embedding_layer = embedding(self.vocab_size,self.embed_size)
        # Create a list of transformer layers
        self.transformer_layers = [transformer(weight_shape=[self.embed_size, self.embed_size], num_heads=self.num_heads) for _ in range(self.num_layers)]
        # Create a linear layer to map the transformer output to logits
        self.logits_layer = dense([self.embed_size,self.vocab_size])
        # Store the parameters of the layers in a list
        self.param = [self.embedding_layer.embeddings]
        for transformer_layer in self.transformer_layers:
            self.param.extend(transformer_layer.weight_list)
        self.param.extend(self.logits_layer.weight_list)
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        x = self.embedding_layer.output(data) # shape: (batch_size, seq_len, embed_size)
        x += positional_encoding(self.max_len,self.embed_size) # shape: (batch_size, seq_len, embed_size)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer.output(x) # shape: (batch_size, seq_len, embed_size)
        output = self.logits_layer.output(x) # shape: (batch_size, seq_len, vocab_size)
        return output
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)