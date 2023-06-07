import tensorflow as tf
import Note.nn.layer.dense as d
from Note.nn.layer.flatten import flatten


class nn:
    def __init__(self):
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    
    
    def build(self):
        self.layer1=d.dense([784,128],activation='relu')
        self.layer2=d.dense([128,10])
        self.param=[self.layer1.weight,self.layer1.bias,self.layer2.weight,self.layer2.bias]
        return
    
    
    def fp(self,data):
        data=flatten(data)
        output1=self.layer1.output(data)
        output2=self.layer2.output(output1)
        return output2
    
    
    def loss(self,output,labels):
        return self.loss_object(labels,output)
