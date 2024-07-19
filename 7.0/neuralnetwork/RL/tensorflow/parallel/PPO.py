import tensorflow as tf
from Note import nn
import gym
from Note.nn.parallel.optimizer import SGD


class actor:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
        self.param=[self.dense1.param,self.dense2.param] # store the network parameters in a list
    
    def fp(self,x):
        x=self.dense1(x)
        return tf.nn.softmax(self.dense2(x))


class critic:
    def __init__(self,state_dim,hidden_dim):
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(1, hidden_dim)
        self.param=[self.dense1.param,self.dense2.param] # store the network parameters in a list
    
    def fp(self,x):
        x=self.dense1(x)
        return self.dense2(x)
    
    
class PPO:
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps,alpha):
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.nn=actor(state_dim,hidden_dim,action_dim)
        nn.assign(self.nn.param,self.actor.param.copy())
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.param=[self.actor.param,self.critic.param]
        self.opt=SGD(param=self.param)
        self.genv=[gym.make('CartPole-v0') for _ in range(5)]
    
    
    def env(self,a=None,p=None,initial=None):
        if initial==True:
            state=self.genv[p].reset(seed=0)
            return state
        else:
            next_state,reward,done,_=self.genv[p].step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        action_prob=tf.gather(self.actor.fp(s),a,axis=1,batch_dims=1)
        action_prob_old=tf.gather(self.nn.fp(s),a,axis=1,batch_dims=1)
        raito=action_prob/action_prob_old
        value=self.critic.fp(s)
        value_tar=tf.cast(r,'float32')+0.98*self.critic.fp(next_s)*(1-tf.cast(d,'float32'))
        TD=value_tar-value
        sur1=raito*TD
        sur2=tf.clip_by_value(raito,clip_value_min=1-self.clip_eps,clip_value_max=1+self.clip_eps)*TD
        clip_loss=-tf.math.minimum(sur1,sur2)
        entropy=action_prob*tf.math.log(action_prob+1e-8)
        clip_loss=clip_loss-self.alpha*entropy
        return [tf.reduce_mean(clip_loss),tf.reduce_mean((TD)**2)]
    
    
    def gradient(self,tape,loss):
        actor_gradient=tape.gradient(loss[0],self.param[0])
        critic_gradient=tape.gradient(loss[1],self.param[1])
        return [actor_gradient,critic_gradient]
    
    
    def opt(self,gradient):
        self.opt.opt(gradient[0],self.param[0])
        self.opt.opt(gradient[1],self.param[1])
        self.actor_old.param=self.actor.param.copy()
        return
    
    
    def update_param(self):
        nn.assign(self.nn.param,self.actor.param.copy())
        return