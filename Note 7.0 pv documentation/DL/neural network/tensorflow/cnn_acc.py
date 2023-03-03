import tensorflow as tf

#An example with accuracy function.
class cnn:
    def __init__(self):
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10)
          ])
        self.param=self.model.weights      #parameter list
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.opt=tf.keras.optimizers.Adam() #optimizer
    
    
    def fp(self,data):         #forward propagation function
        output=self.model(data)
        return output
    
    
    def loss(self,output,labels): #loss functino
        return self.loss_object(labels,output)
    
    
    def accuracy(self,output,labels): #accuracy function
        return self.train_accuracy(labels,output)
