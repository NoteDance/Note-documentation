import torch
import torch.nn.functional as F
import numpy as np


class actor(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound):
        super(actor,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
        self.action_bound=action_bound
    
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))*self.action_bound


class critic(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(critic,self).__init__()
        self.fc1=torch.nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)
    
    
    def forward(self,x,a):
        cat=torch.cat([x,a],dim=1)
        x=F.relu(self.fc1(cat))
        return self.fc2(x)


class DDPG:
    def __init__(self,state_dim,hidden_dim,action_dim,action_bound,sigma,gamma,tau,actor_lr,critic_lr):
        if torch.cuda.is_available():
            self.device_d=torch.device('cuda')
            self.device_n=torch.device('cuda')
        else:
            self.device_d=torch.device('cpu')
            self.device_n=torch.device('cpu')
        self.actor=actor(state_dim,hidden_dim,action_dim,action_bound).to(self.device_n)
        self.critic=critic(state_dim,hidden_dim,action_dim).to(self.device_n)
        self.target_actor=actor(state_dim,hidden_dim,action_dim,action_bound).to(self.device_n)
        self.target_critic=critic(state_dim,hidden_dim,action_dim).to(self.device_n)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.sigma=sigma
        self.gamma=gamma
        self.tau=tau
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.genv=None
    
    
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
        s=torch.tensor(s,dtype=torch.float).to(self.device_d)
        a=torch.tensor(a,dtype=torch.float).view(-1,1).to(self.device_d)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device_d)
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(self.device_d)
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(self.device_d)
        next_q_value=self.target_critic(torch.cat([next_s,self.target_actor(next_s)],dim=1))
        q_target=r+self.gamma*next_q_value*(1-d)
        actor_loss=-torch.mean(self.critic(torch.cat([s,self.actor(s)],dim=1)))
        critic_loss=F.mse_loss(self.critic(torch.cat([s,a],dim=1)),q_target)
        return [actor_loss,critic_loss]
    
    
    def opt(self,loss):
        self.actor_opt.zero_grad()
        loss[0].backward()
        self.actor_opt.step()
        self.critic_opt.zero_grad()
        loss[1].backward()
        self.critic_opt.step()
        return
        
    
    def update_param(self):
        for target_param,param in zip(self.target_actor.parameters(),self.actor.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.tau)+param.data*self.tau)
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.tau)+param.data*self.tau)
        return
