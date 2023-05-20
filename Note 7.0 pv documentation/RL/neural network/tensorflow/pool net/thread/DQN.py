import tensorflow as tf
import gym


class Qnet:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.weight1=tf.Variable(tf.random.normal([state_dim,hidden_dim]))
        self.bias1=tf.Variable(tf.random.normal([hidden_dim]))
        self.weight2=tf.Variable(tf.random.normal([hidden_dim,action_dim]))
        self.bias2=tf.Variable(tf.random.normal([action_dim]))
        self.param=[self.weight1,self.bias1,self.weight2,self.bias2]    
    
    def fp(self,x):  #forward propagation function,kernel uses it for forward propagation.
        x=tf.nn.relu(tf.matmul(x,self.weight1)+self.bias1)
        return tf.matmul(x,self.weight2)+self.bias2
    
    
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.nn=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.nn.param   #parameter list,kernel uses it list for backpropagation.
        self.opt=tf.keras.optimizers.Adam() #optimizer,kernel uses it to optimize.
        self.genv=[gym.make('CartPole-v0') for _ in range(5)] #create environment.
    
    
    def env(self,a=None,t=None,initial=None): #environment function,kernel uses it to interact with the environment.
        if initial==True:
            state=self.genv[t].reset(seed=0)
            return state
        else:
            next_state,reward,done,_=self.genv[t].step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d): #loss functino,kernel uses it to calculate loss.
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.nn.fp(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net.fp(next_s),axis=1)
        target=r+0.98*next_q_value*(1-d)
        return tf.reduce_mean((q_value-target)**2)
        
    
    def update_param(self): #update function,kernel uses it to update parameter.
        self.target_q_net.param=self.param.copy()
        return
