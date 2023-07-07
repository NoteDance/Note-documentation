import tensorflow as tf
import numpy as np

#This is an example of online training.
class nn:
    def __init__(self,data,labels):
        self.train_data=data
        self.train_labels=labels
        self.model=tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128,activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10)
          ])
        self.param=self.model.weights #parameter list,kernel uses it list for backpropagation.
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam() #optimizer,kernel uses it to optimize.
        self.train_loss_list=[]
        self.c=0 #counter
        self.max_length=1000
    
    
    def ol(self): #This is simulative online function.
        index=np.random.choice(60000,size=[32])
        if self.c==10000:
            return 'stop'
        else:
            return [self.train_data[index],self.train_labels[index]]
    
    
    def fp(self,data):  #forward propagation function,kernel uses it for forward propagation.
        output=self.model(data)
        return output
    
    
    def loss(self,output,labels): #loss functino,kernel uses it to calculate loss.
        return self.loss_object(labels,output)