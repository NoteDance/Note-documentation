import tensorflow as tf
from tensorflow.python.ops import state_ops


class Adam: #You can also realize optimizer by yourself
    def __init__(self,lr=0.007,beta1=0.9,beta2=0.999,epsilon=1e-07):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.v=[]
        self.s=[]
        self.v_=[]
        self.s_=[]
        self.g=[]
        self.flag=0
    
    
    def opt(self,gradient,parameter,t):
        if self.flag==0:
            self.v=[0 for x in range(len(gradient))]
            self.s=[0 for x in range(len(gradient))]
            self.v_=[x for x in range(len(gradient))]
            self.s_=[x for x in range(len(gradient))]
            self.g=[x for x in range(len(gradient))]
            self.flag+=1
        for i in range(len(gradient)):
            self.v[i]=self.beta1*self.v[i]+(1-self.beta1)*gradient[i]
            self.s[i]=self.beta2*self.s[i]+(1-self.beta2)*gradient[i]**2
            self.v_[i]=self.v[i]/(1-self.beta1**(t+1))
            self.s_[i]=self.s[i]/(1-self.beta2**(t+1))
            self.g[i]=self.lr*self.v_[i]/(tf.sqrt(self.s_[i])+self.epsilon)
            state_ops.assign(parameter[i],parameter[i]-self.g[i])
        return


class nn:               #A neural network class example,use optimizer written by yourself.
    def __init__(self):
        self.weight1=tf.Variable(tf.random.normal([784,64],dtype='float64')) #You can also customize parameters.
        self.bias1=tf.Variable(tf.random.normal([64],dtype='float64'))
        self.weight2=tf.Variable(tf.random.normal([64,64],dtype='float64'))
        self.bias2=tf.Variable(tf.random.normal([64],dtype='float64'))
        self.weight3=tf.Variable(tf.random.normal([64,10],dtype='float64'))
        self.bias3=tf.Variable(tf.random.normal([10],dtype='float64'))
        self.param=[self.weight1,self.weight2,self.weight3,self.bias1,self.bias2,self.bias3] #parameter list,kernel uses it list for backpropagation.
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=Adam() #optimizer,kernel uses it to optimize.
        self.bc=0 #batch counter,kernel use it to count.
        self.info='example'
    
    
    def fp(self,data):  #forward propagation function,kernel uses it for forward propagation.
        layer1=tf.nn.relu(tf.matmul(data,self.param[0])+self.param[3])
        layer2=tf.nn.relu(tf.matmul(layer1,self.param[1])+self.param[4])
        output=tf.matmul(layer2,self.param[2])+self.param[5]
        return output
    
    
    def loss(self,output,labels): #loss functino,kernel uses it to calculate loss.
        return self.loss_object(labels,output)
    

    def opt(self,gradient,param): #optimization function,kernel uses it to optimize.
        self.optimizer.opt(gradient,param,self.bc)
        return
