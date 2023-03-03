import tensorflow as tf


class cnn:
    def __init__(self):
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10)
          ])
        self.param=self.model.weights #parameter list,kernel uses this list for backpropagation.
        self.opt=tf.keras.optimizers.Adam() #optimizer,kernel uses this to optimize.
    
    
    def fp(self,data):  #forward propagation function,kernel uses this for forward propagation.
        output=self.model(data)
        return output
    
    
    def loss(self,output,labels): #loss functino,kernel uses this to calculate loss.
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels))
