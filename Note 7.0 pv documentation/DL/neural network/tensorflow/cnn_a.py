import tensorflow as tf


#gradient attenuation example
class cnn:
    def __init__(self):
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10)
          ])
        self.param=self.model.weights
        self.opt=tf.keras.optimizers.Adam(learning_rate=0.001)
    
    
    def fp(self,data):
        output=self.model(data)
        return output
    
    
    def loss(self,output,labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels))
    
    
    def attenuate(self,gradient,oc,t):  #gradient attenuation function,kernel uses it to calculate attenuation coefficient.
        ac=0.9**oc[t]                   #ac:attenuation coefficient
        for i in range(len(gradient)):  #oc:optimizing counter
            gradient[i]=ac*gradient[i]  #t:thread number
        return gradient                              
