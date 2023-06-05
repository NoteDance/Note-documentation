import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.flatten import flatten

# Define a convolutional neural network class
class cnn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    
    
    def build(self):
        # Create three convolutional layers with relu activations
        self.conv1=conv2d(weight_shape=(3,3,3,32),activation='relu')
        self.conv2=conv2d(weight_shape=(3,3,32,64),activation='relu')
        self.conv3=conv2d(weight_shape=(3,3,64,64),activation='relu')
        # Create two dense layers with relu and linear activations
        self.dense1=dense([64*4*4,64],activation='relu')
        self.dense2=dense([64,10])
        # Store the parameters of the layers in a list
        self.param = [self.conv1.weight,
                      self.conv2.weight,
                      self.conv3.weight,
                      self.dense1.weight,
                      self.dense1.bias,
                      self.dense2.weight,
                      self.dense2.bias]
        return
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        x=tf.nn.conv2d(data,self.conv1.weight,strides=(1,1),padding='SAME') # First convolutional layer
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID') # First max pooling layer
        x=tf.nn.conv2d(x,self.conv2.weight,strides=(1,1),padding='SAME') # Second convolutional layer
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID') # Second max pooling layer
        x=tf.nn.conv2d(x,self.conv3.weight,strides=(1,1),padding='SAME') # Third convolutional layer
        x=flatten(x) # Flatten the output to a vector
        x=tf.nn.relu(tf.matmul(x,self.dense1.weight)+self.dense1.bias) # First dense layer with relu activation
        x=tf.nn.dropout(x,rate=0.5) # Apply dropout to prevent overfitting
        output=tf.matmul(x,self.dense2.weight)+self.dense2.bias # Output layer with linear activation
        output=tf.nn.dropout(output,rate=0.5) # Apply dropout to prevent overfitting
        return output
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)