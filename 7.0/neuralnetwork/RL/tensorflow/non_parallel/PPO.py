import tensorflow as tf
import gym


class actor:
    def __init__(self,state_dim,hidden_dim,action_dim):
        self.weight1=tf.Variable(tf.random.normal([state_dim,hidden_dim]))
        self.bias1=tf.Variable(tf.random.normal([hidden_dim]))
        self.weight2=tf.Variable(tf.random.normal([hidden_dim,action_dim]))
        self.bias2=tf.Variable(tf.random.normal([action_dim]))
        self.param=[self.weight1,self.bias1,self.weight2,self.bias2]
    
    def fp(self,x):
        x=tf.nn.relu(tf.matmul(x,self.weight1)+self.bias1)
        return tf.nn.softmax(tf.matmul(x,self.weight2)+self.bias2)


class critic:
    def __init__(self,state_dim,hidden_dim):
        self.weight1=tf.Variable(tf.random.normal([state_dim,hidden_dim]))
        self.bias1=tf.Variable(tf.random.normal([hidden_dim]))
        self.weight2=tf.Variable(tf.random.normal([hidden_dim,1]))
        self.bias2=tf.Variable(tf.random.normal([1]))
        self.param=[self.weight1,self.bias1,self.weight2,self.bias2]
    
    def fp(self,x):
        x=tf.nn.relu(tf.matmul(x,self.weight1)+self.bias1)
        return tf.matmul(x,self.weight2)+self.bias2
    
    
class PPO:
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps):
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.nn=actor(state_dim,hidden_dim,action_dim)
        self.nn.param=self.actor.param.copy()
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.param=[self.actor.param,self.critic.param]
        self.opt=tf.keras.optimizers.Adam()
        self.genv=gym.make('CartPole-v0')
    
    
    def env(self,a=None,initial=None):
        if initial==True:
            state=self.genv.reset(seed=0)
            return state
        else:
            next_state,reward,done,_=self.genv.step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        raito=tf.gather(self.actor.fp(s),a,axis=1,batch_dims=1)/tf.gather(self.nn.fp(s),a,axis=1,batch_dims=1)
        value=self.critic.fp(s)
        value_tar=r+self.critic.fp(next_s)
        TD=r+0.98*value_tar*(1-d)-value
        sur1=raito*TD
        sur2=tf.clip_by_value(raito,clip_value_min=1-self.clip_eps,clip_value_max=1+self.clip_eps)*TD
        clip_loss=-tf.math.minimum(sur1,sur2)
        return [tf.reduce_mean(clip_loss),tf.reduce_mean((TD)**2)]
    
    
    def gradient(self,tape,loss):
        actor_gradient=tape.gradient(loss[0],self.param[0])
        critic_gradient=tape.gradient(loss[1],self.param[1])
        return [actor_gradient,critic_gradient]
    
    
    def opt(self,gradient):
        self.opt.apply_gradients(zip(gradient[0],self.param[0]))
        self.opt.apply_gradients(zip(gradient[1],self.param[1]))
        self.actor_old.param=self.actor.param.copy()
        return
    
    
    def update_param(self):
        self.nn.param=self.actor.param.copy()
        return
