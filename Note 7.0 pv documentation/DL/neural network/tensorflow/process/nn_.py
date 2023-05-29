import tensorflow as tf
import Note.nn.process.optimizer as o


class nn:
    def __init__(self):
        self.weight1=tf.Variable(tf.random.normal([784,64],dtype='float64'))
        self.bias1=tf.Variable(tf.random.normal([64],dtype='float64'))
        self.weight2=tf.Variable(tf.random.normal([64,64],dtype='float64'))
        self.bias2=tf.Variable(tf.random.normal([64],dtype='float64'))
        self.weight3=tf.Variable(tf.random.normal([64,10],dtype='float64'))
        self.bias3=tf.Variable(tf.random.normal([10],dtype='float64'))
        self.param=[self.weight1,self.weight2,self.weight3,self.bias1,self.bias2,self.bias3]
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=o.Momentum(0.07,0.7)
    
    
    def fp(self,data):
        layer1=tf.nn.relu(tf.matmul(data,self.weight1)+self.bias1)
        layer2=tf.nn.relu(tf.matmul(layer1,self.weight2)+self.bias2)
        output=tf.matmul(layer2,self.weight3)+self.bias3
        return output
    
    
    def loss(self,output,labels):
        return self.loss_object(labels,output)
    

    def opt(self,gradient):
        param=self.optimizer.opt(gradient,self.param)
        return param
