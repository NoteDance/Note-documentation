import tensorflow as tf
from tensorflow.keras import layers,models

#https://tensorflow.google.cn/tutorials/images/cnn
class cifar:
    def __init__(self):
        self.model=models.Sequential()
        self.model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64,(3,3),activation='relu'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64,(3,3),activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64,activation='relu'))
        self.model.add(layers.Dense(10))
        self.param=self.model.weights
        self.opt=tf.keras.optimizers.Adam()
    
    
    def fp(self,data):
        with tf.device('GPU:0'):
            output=self.model(data)
        return output
    
    
    def loss(self,output,labels):
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss(labels,output)