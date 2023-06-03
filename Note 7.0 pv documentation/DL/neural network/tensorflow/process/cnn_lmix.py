import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.flatten import flatten
import Note.nn.process.optimizer as o
from Note.nn.layer.LMix import lmix


class cnn:
    def __init__(self):
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=o.Adam()
        self.alpha=1.0
        self.bc=tf.Variable(0,dtype=tf.float64)
    
    
    def build(self):
        self.conv1=conv2d(weight_shape=(3,3,3,32),activation='relu')
        self.conv2=conv2d(weight_shape=(3,3,32,64),activation='relu')
        self.conv3=conv2d(weight_shape=(3,3,64,64),activation='relu')
        self.dense1=dense([64*4*4,64],activation='relu')
        self.dense2=dense([64,10])
        self.param = [self.conv1.weight,
                      self.conv2.weight,
                      self.conv3.weight,
                      self.dense1.weight,
                      self.dense1.bias,
                      self.dense2.weight,
                      self.dense2.bias]
        return
    
    
    def data_func(self,data_batch,labels_batch):
        return lmix(data_batch,labels_batch,self.alpha,128)
    
    
    def fp(self,data):
        x=tf.nn.conv2d(data,self.conv1.weight,strides=(1,1),padding='SAME')
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID')
        x=tf.nn.conv2d(x,self.conv2.weight,strides=(1,1),padding='SAME')
        x=tf.nn.max_pool2d(x,ksize=(2,2),strides=(2,2),padding='VALID')
        x=tf.nn.conv2d(x,self.conv3.weight,strides=(1,1),padding='SAME')
        x=flatten(x)
        x=tf.nn.relu(tf.matmul(x,self.dense1.weight)+self.dense1.bias)
        x=tf.nn.dropout(x,rate=0.5)
        output=tf.matmul(x,self.dense2.weight)+self.dense2.bias
        output=tf.nn.dropout(output,rate=0.5)
        return output
    
    
    def loss(self,output,labels):
        return self.loss_object(labels,output)
    
    
    def opt(self,gradient):
        param=self.optimizer.opt(gradient,self.param,self.bc)
        return param