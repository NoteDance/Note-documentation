import tensorflow as tf


class cnn:
    def __init__(self):
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10,activation='softmax')
          ])
        self.param=self.model.weights
        self.opt=tf.keras.optimizers.Adam()
    
    
    @tf.function
    def fp(self,data):
        return self.model(data)
    
    
    def loss(self,output,labels):
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss(output,labels)