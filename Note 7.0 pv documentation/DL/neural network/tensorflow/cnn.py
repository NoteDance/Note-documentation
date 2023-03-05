import tensorflow as tf


class cnn:
    def __init__(self):
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10)
          ])
        self.param=self.model.weights #parameter list,kernel uses it list for backpropagation.
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam() #optimizer,kernel uses it to optimize.
    
    
    def fp(self,data):  #forward propagation function,kernel uses it for forward propagation.
        output=self.model(data)
        return output
    
    
    def loss(self,output,labels): #loss functino,kernel uses it to calculate loss.
        return self.loss_object(labels,output)
