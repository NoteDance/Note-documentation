import tensorflow as tf
import numpy as np
import gym


class actor:
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound):
        self.weight1=tf.Variable(tf.random.normal([state_dim,hidden_dim]))
        self.bias1=tf.Variable(tf.random.normal([hidden_dim]))
        self.weight2=tf.Variable(tf.random.normal([hidden_dim,action_dim]))
        self.bias2=tf.Variable(tf.random.normal([action_dim]))
        self.action_bound=action_bound
        self.param=[self.weight1,self.bias1,self.weight2,self.bias2]  
    
    
    def fp(self,x):
        x=tf.nn.relu(tf.matmul(x,self.weight1)+self.bias1)
        return tf.nn.tanh(tf.matmul(x,self.weight2)+self.bias2)*self.action_bound


class critic:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.weight1=tf.Variable(tf.random.normal([state_dim+action_dim,hidden_dim]))
        self.bias1=tf.Variable(tf.random.normal([hidden_dim]))
        self.weight2=tf.Variable(tf.random.normal([hidden_dim,action_dim]))
        self.bias2=tf.Variable(tf.random.normal([action_dim]))
        self.param=[self.weight1,self.bias1,self.weight2,self.bias2]
    
    
    def fp(self,x,a):
        cat=tf.concat([x,a],axis=1)
        x=tf.nn.relu(tf.matmul(cat,self.weight1)+self.bias1)
        return tf.matmul(x,self.weight2)+self.bias2


class DDPG:
    def __init__(self,hidden_dim,sigma,gamma,tau,actor_lr,critic_lr):
        self.genv=gym.make('Pendulum-v0')
        state_dim=self.genv.observation_space.shape[0]
        action_dim=self.genv.action_space.shape[0]
        action_bound=self.genv.action_space.high[0]
        self.actor=actor(state_dim,hidden_dim,action_dim,action_bound)
        self.critic=critic(state_dim,hidden_dim,action_dim)
        self.target_actor=actor(state_dim,hidden_dim,action_dim,action_bound)
        self.target_critic=critic(state_dim,hidden_dim,action_dim)
        self.actor_param=self.actor.param
        self.critic_param=self.critic.param
        self.target_actor.param=self.actor_param.copy()
        self.target_critic.param=self.critic_param.copy()
        self.param=[self.actor_param,self.critic_param]
        self.sigma=sigma
        self.gamma=gamma
        self.tau=tau
        self.opt=tf.keras.optimizers.Adam()
    
    
    def noise(self):
        return np.random.normal(scale=self.sigma)
    
    
    def env(self,a=None,initial=None):
        if initial==True:
            state=self.genv.reset(seed=0)
            return state
        else:
            next_state,reward,done,_=self.genv.step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        next_q_value=self.target_critic.fp(next_s,self.target_actor.fp(next_s))
        q_target=r+self.gamma*next_q_value*(1-d)
        actor_loss=-tf.reduce_mean(self.critic.fp(s,self.actor.fp(s)))
        critic_loss=tf.reduce_mean((self.critic.fp(s,a)-q_target)**2)
        return [actor_loss,critic_loss]
        
    
    def update_param(self):
        for target_param,param in zip(self.target_actor.param,self.actor.param):
            target_param=target_param*(1.0-self.tau)+param*self.tau
        for target_param,param in zip(self.target_critic.param,self.critic.param):
            target_param=target_param*(1.0-self.tau)+param*self.tau
        return
