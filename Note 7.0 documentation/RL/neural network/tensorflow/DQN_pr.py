import tensorflow as tf
import gym
import Note.create.RL.rl.prioritized_replay as pr


class Qnet:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.weight1=tf.Variable(tf.random.normal([state_dim,hidden_dim]))
        self.bias1=tf.Variable(tf.random.normal([hidden_dim]))
        self.weight2=tf.Variable(tf.random.normal([hidden_dim,action_dim]))
        self.bias2=tf.Variable(tf.random.normal([action_dim]))
        self.param=[self.weight1,self.bias1,self.weight2,self.bias2]    
    
    def fp(self,x):
        x=tf.nn.relu(tf.matmul(x,self.weight1)+self.bias1)
        return tf.matmul(x,self.weight2)+self.bias2
    
    
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.nn=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.nn.param
        self.pr=pr.pr()
        self.initial_TD=7
        self._epsilon=0.0007
        self.alpha=0.7
        self.opt=tf.keras.optimizers.Adam()
        self.genv=gym.make('CartPole-v0')
    
    
    def env(self,a=None,initial=None):
        if initial==True:
            state=self.genv.reset(seed=0)
            return state
        else:
            next_state,reward,done,_=self.genv.step(a)
            return next_state,reward,done
    
    
    def data_func(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,batch):
        s,a,next_s,r,d=self.pr.sample(state_pool,action_pool,next_state_pool,reward_pool,done_pool,self.epsilon,self.alpha,batch)
        return s,a,next_s,r,d
    
    
    def loss(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        q_value=tf.gather(self.nn.fp(s),a,axis=1,batch_dims=1)
        next_q_value=tf.reduce_max(self.target_q_net.fp(next_s),axis=1)
        target=r+0.98*next_q_value*(1-d)
        target=tf.expand_dims(target,axis=1)
        TD=target-q_value
        self.pr.update_TD(TD)
        return tf.reduce_mean(TD**2)
        
    
    def update_param(self):
        self.target_q_net.param=self.param.copy()
        return