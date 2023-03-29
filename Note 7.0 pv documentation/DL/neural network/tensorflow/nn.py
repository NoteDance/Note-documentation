import tensorflow as tf
from tensorflow.python.ops import state_ops


class Momentum: #You can also realize optimizer by yourself
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
        return


class nn:               #A neural network class example,use optimizer written by yourself.
    def __init__(self):
        self.weight1=tf.Variable(tf.random.normal([784,64])) #You can also customize parameters.
        self.bias1=tf.Variable(tf.random.normal([64]))
        self.weight2=tf.Variable(tf.random.normal([64,64]))
        self.bias2=tf.Variable(tf.random.normal([64]))
        self.weight3=tf.Variable(tf.random.normal([64,10]))
        self.bias3=tf.Variable(tf.random.normal([10]))
        self.param=[self.weight1,self.weight2,self.weight3,self.bias1,self.bias2,self.bias3] #parameter list,kernel uses it list for backpropagation.
        self.optimizer=Momentum(0.07,0.7) #optimizer,kernel uses it to optimize.
        self.info='example'
    
    
    def fp(self,data):  #forward propagation function,kernel uses it for forward propagation.
        layer1=tf.nn.relu(tf.matmul(data,self.param[0])+self.param[3])
        layer2=tf.nn.relu(tf.matmul(layer1,self.param[1])+self.param[4])
        output=tf.matmul(layer2,self.param[2])+self.param[5]
        return output
    
    
    def loss(self,output,labels): #loss functino,kernel uses it to calculate loss.
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels))
    

    def opt(self,gradient,param): #optimization function,kernel uses it to optimize.
        self.optimizer.opt(gradient,param)
        return
