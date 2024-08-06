import tensorflow as tf # import TensorFlow library
from Note import nn
import gym # import OpenAI Gym library


class Qnet: # define a class for the Q-network
    def __init__(self,state_dim,hidden_dim,action_dim): # initialize the network with state dimension, hidden dimension and action dimension
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
        self.param=[self.dense1.param,self.dense2.param] # store the network parameters in a list
    
    def fp(self,x):  # forward propagation function, kernel uses it for forward propagation
        x = self.dense2(self.dense1(x))
        return x
    
    
class DQN: # define a class for the DQN agent
    def __init__(self,state_dim,hidden_dim,action_dim): # initialize the agent with state dimension, hidden dimension and action dimension
        self.nn=Qnet(state_dim,hidden_dim,action_dim) # create a Q-network for the agent
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim) # create a target Q-network for the agent
        self.param=self.nn.param   # parameter list, kernel uses it list for backpropagation
        self.opt=tf.keras.optimizers.Adam() # optimizer, kernel uses it to optimize
        self.genv=gym.make('CartPole-v0') # create environment
    
    
    def env(self,a=None,initial=None): # environment function, kernel uses it to interact with the environment
        if initial==True: # if initial is True, reset the environment and return the initial state
            state=self.genv.reset(seed=0)
            return state
        else: # otherwise, take an action and return the next state, reward and done flag
            next_state,reward,done,_=self.genv.step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d): # loss function, kernel uses it to calculate loss
        a=tf.expand_dims(a,axis=1) # expand the action vector to match the Q-value matrix shape
        q_value=tf.gather(self.nn.fp(s),a,axis=1,batch_dims=1) # get the Q-value for the selected action
        next_q_value=tf.reduce_max(self.target_q_net.fp(next_s),axis=1) # get the maximum Q-value for the next state from the target network
        target=tf.cast(r,'float32')+0.98*next_q_value*(1-tf.cast(d,'float32')) # calculate the target value using Bellman equation with discount factor 0.98
        return tf.reduce_mean((q_value-target)**2) # return the mean squared error between Q-value and target value
        
    
    def update_param(self): # update function, kernel uses it to update parameter
        nn.assign_param(self.target_q_net.param, self.param) # copy the parameters from the Q-network to the target network
        return