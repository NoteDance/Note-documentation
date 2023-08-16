import tensorflow as tf # import TensorFlow library
import numpy as np # import NumPy library

# This is an example of online training.
class nn: # define a class for the neural network
    def __init__(self,data,labels): # initialize the network with data and labels
        self.train_data=data # store the training data
        self.train_labels=labels # store the training labels
        self.model=tf.keras.models.Sequential([ # create a sequential model with four layers
          tf.keras.layers.Flatten(input_shape=(28, 28)), # flatten the input to a vector of 28*28 elements
          tf.keras.layers.Dense(128,activation='relu'), # add a dense layer with 128 units and ReLU activation
          tf.keras.layers.Dropout(0.2), # add a dropout layer with 0.2 dropout rate to prevent overfitting
          tf.keras.layers.Dense(10) # add a dense layer with 10 units for the output logits
          ])
        self.param=self.model.weights # parameter list, kernel uses it list for backpropagation
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss object, kernel uses it to calculate loss. Here we use sparse categorical crossentropy with logits as output
        self.opt=tf.keras.optimizers.Adam() # optimizer, kernel uses it to optimize. Here we use Adam optimizer
        self.train_loss_list=[] # a list to store the training loss values
        self.counter=0 # counter to keep track of the number of online updates
        self.max_length=1000 # maximum length of train_loss_list
    
    
    def online(self): # This is simulative online function, kernel uses it to get online data and labels
        index=np.random.choice(60000,size=[32]) # randomly sample 32 indices from 0 to 59999
        if self.counter==10000: # if the counter reaches 10000, stop the online training
            return 'stop'
        else: # otherwise, return the data and labels corresponding to the sampled indices
            return [self.train_data[index],self.train_labels[index]]
    
    
    def fp(self,data):  # forward propagation function, kernel uses it for forward propagation
        output=self.model(data) # pass the data through the model and get the output logits
        return output
    
    
    def loss(self,output,labels): # loss function, kernel uses it to calculate loss
        return self.loss_object(labels,output) # return the sparse categorical crossentropy loss between labels and output