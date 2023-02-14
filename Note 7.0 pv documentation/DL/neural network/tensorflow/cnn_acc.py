import tensorflow as tf


class cnn:
    def __init__(self):
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10)
          ])
        self.param=self.model.weights
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.opt=tf.keras.optimizers.Adam()
    
    
    def fp(self,data):
        output=self.model(data)
        return output
    
    
    def loss(self,output,labels):
        return self.loss_object(labels,output)
    
    
    def accuracy(self,output,labels):
        return self.train_accuracy(labels,output)