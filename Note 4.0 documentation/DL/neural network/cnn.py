import tensorflow as tf

#https://tensorflow.google.cn/tutorials/quickstart/beginner
class cnn:
    def __init__(self):
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10)
          ])
        self.param=self.model.weights
        self.opt=tf.keras.optimizers.Adam()
    
    
    def fp(self,data):
        with tf.device('GPU:0'):
            output=self.model(data)
        return output
    
    
    def loss(self,output,labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels))        