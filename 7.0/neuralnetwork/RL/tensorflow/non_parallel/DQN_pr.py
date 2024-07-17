import tensorflow as tf
from Note import nn
import gym
import Note.RL.rl.prioritized_replay as pr


class Qnet:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
        self.param=[self.dense1.param,self.dense2.param] # store the network parameters in a list   
    
    def fp(self,x):
        x = self.dense2(self.dense1(x))
        return x
    
    
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.nn=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.param=self.nn.param
        self.pr=pr.pr()
        self.initial_TD=7
        self.epsilon=0.0007
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
        target=r+0.98*next_q_value*(1-tf.cast(d,'float32'))
        target=tf.expand_dims(target,axis=1)
        TD=target-q_value
        self.pr.update_TD(TD)
        return tf.reduce_mean(TD**2)
        
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param, self.param.copy())
        return