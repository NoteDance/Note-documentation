import torch
import gym
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)
    
    
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,device):
        self.nn=Qnet(state_dim,hidden_dim,action_dim).to(device)
        self.target_q_net(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=2e-3)
        self.env=gym.make('CartPole-v0')
        self.device=device
    
    
    def explore(self,a=None,init=None):
        if init==True:
            self.env.action_space.seed(0)
            state,info=self.env.reset(seed=0,return_info=True)
            return state
        else:
            next_state,reward,done,_=self.env.step(a)
            if done:
                next_state,info=self.env.reset(return_info=True)
            return next_state,reward,done
    
    
    def loss(self,s_batch,a_batch,next_s_batch,r_batch,d_batch):
        s_batch=torch.tensor(s_batch,dtype=torch.float).to(self.device)
        a_batch=torch.tensor(a_batch,dtype=torch.float).view(-1,1).to(self.device)
        next_s_batch=torch.tensor(next_s_batch,dtype=torch.float).to(self.device)
        r_batch=torch.tensor(r_batch,dtype=torch.float.view(-1,1)).to(self.device)
        d_batch=torch.tensor(d_batch,dtype=torch.float.view(-1,1)).to(self.device)
        q_value=self.nn(s_batch).gather(1,a_batch)
        next_q_value=self.target_q_net(next_s_batch).max(1)[0].view(-1,1)
        target=r_batch+0.98*next_q_value*(1-d_batch)
        return torch.mean(F.mse_loss(q_value,target))
    
    
    def opt(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def update_param(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())