import tensorflow as tf # import TensorFlow library
from Note import nn
import numpy as np # import NumPy library
import gym # import OpenAI Gym library


class actor: # define a class for the actor network
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound): # initialize the network with state dimension, hidden dimension, action dimension and action bound
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim, activation='tanh')
        self.action_bound=action_bound
        self.param=[self.dense1.param,self.dense2.param] # store the network parameters in a list
    
    
    def fp(self,x):  # forward propagation function, kernel uses it for forward propagation
        x = self.dense1(x)
        return self.dense2(x)*self.action_bound


class critic: # define a class for the critic network
    def __init__(self,state_dim,hidden_dim,action_dim): # initialize the network with state dimension, hidden dimension and action dimension
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
        self.param=[self.dense1.param,self.dense2.param] # store the network parameters in a list
    
    
    def fp(self,x,a):  # forward propagation function, kernel uses it for forward propagation
        cat=tf.concat([x,a],axis=1)
        x=self.dense1(cat)
        return self.dense2(x)


class DDPG: # define a class for the DDPG agent
    def __init__(self,hidden_dim,sigma,gamma,tau,actor_lr,critic_lr): # initialize the agent with hidden dimension, noise scale, discount factor, soft update factor, actor learning rate and critic learning rate
        self.genv=gym.make('Pendulum-v0') # create environment
        state_dim=self.genv.observation_space.shape[0] # get the state dimension from the environment observation space
        action_dim=self.genv.action_space.shape[0] # get the action dimension from the environment action space
        action_bound=self.genv.action_space.high[0] # get the action bound from the environment action space high value (assume symmetric)
        self.actor=actor(state_dim,hidden_dim,action_dim,action_bound) # create an actor network for the agent
        self.critic=critic(state_dim,hidden_dim,action_dim) # create a critic network for the agent
        self.target_actor=actor(state_dim,hidden_dim,action_dim,action_bound) # create a target actor network for the agent
        self.target_critic=critic(state_dim,hidden_dim,action_dim) # create a target critic network for the agent
        self.actor_param=self.actor.param  # parameter list of actor network, kernel uses it list for backpropagation 
        self.critic_param=self.critic.param  # parameter list of critic network, kernel uses it list for backpropagation 
        nn.assign_param(self.target_actor.param,self.actor_param.copy())  # copy the parameters from actor network to target actor network 
        nn.assign_param(self.target_critic.param,self.critic_param.copy())  # copy the parameters from critic network to target critic network 
        self.param=[self.actor_param,self.critic_param]  # parameter list of both networks, kernel uses it list for backpropagation 
        self.sigma=sigma  # noise scale 
        self.gamma=gamma  # discount factor 
        self.tau=tau  # soft update factor 
        self.opt=tf.keras.optimizers.Adam()  # optimizer, kernel uses it to optimize. Here we use Adam optimizer
    
    
    def noise(self):  # noise function, kernel uses it to generate exploration noise
        return np.random.normal(scale=self.sigma)  # return a random sample from a normal distribution with zero mean and sigma scale
    
    
    def env(self,a=None,initial=None):  # environment function, kernel uses it to interact with the environment
        if initial==True:  # if initial is True, reset the environment and return the initial state
            state=self.genv.reset(seed=0)
            return state
        else:  # otherwise, take an action and return the next state, reward and done flag
            next_state,reward,done,_=self.genv.step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d):  # loss function, kernel uses it to calculate loss
        a=tf.expand_dims(a,axis=1)  # expand the action vector to match the Q-value matrix shape
        next_q_value=self.target_critic.fp(next_s,self.target_actor.fp(next_s))  # get the Q-value for the next state and action from the target critic network
        q_target=r+self.gamma*next_q_value*(1-d)  # calculate the target value using Bellman equation with discount factor gamma
        actor_loss=-tf.reduce_mean(self.critic.fp(s,self.actor.fp(s)))  # calculate the actor loss as the negative mean of the Q-value for the current state and action from the critic network
        critic_loss=tf.reduce_mean((self.critic.fp(s,a)-q_target)**2)  # calculate the critic loss as the mean squared error between Q-value and target value
        return [actor_loss,critic_loss]  # return a list of actor loss and critic loss
        
    
    def update_param(self):  # update function, kernel uses it to update parameter
        for target_param,param in zip(self.target_actor.param,self.actor.param):  # for each pair of parameters in target actor network and actor network
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)  # update the target parameter using soft update with factor tau
        for target_param,param in zip(self.target_critic.param,self.critic.param):  # for each pair of parameters in target critic network and critic network
            target_param.assign(target_param*(1.0-self.tau)+param*self.tau)  # update the target parameter using soft update with factor tau
        return
