import tensorflow as tf
from tensorflow.python.ops import state_ops


class Momentum:
    def __init__(self,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.v=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter):
        if self.flag==0:
            self.v=[0 for x in range(len(gradient))]
        for i in range(len(gradient)):
            self.v[i]=self.gamma*self.v[i]+self.lr*gradient[i]
            state_ops.assign(parameter[i],parameter[i]-self.v[i])
        return parameter


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
        self.optimizer=Momentum(0.07,0.7)
    
    
    def fp(self,data,param=None):
        if param==None:
            layer1=tf.nn.relu(tf.matmul(data,self.weight1)+self.bias1)
            layer2=tf.nn.relu(tf.matmul(layer1,self.weight2)+self.bias2)
            output=tf.matmul(layer2,self.weight3)+self.bias3
        else:
            layer1=tf.nn.relu(tf.matmul(data,param[0])+param[3])
            layer2=tf.nn.relu(tf.matmul(layer1,param[1])+param[4])
            output=tf.matmul(layer2,param[2])+param[5]
        return output
    
    
    def loss(self,output,labels):
        return self.loss_object(labels,output)
    

    def opt(self,gradient):
        param=self.optimizer.opt(gradient,self.param)
        return param