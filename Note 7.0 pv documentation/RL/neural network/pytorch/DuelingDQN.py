import torch
import gym
import torch.nn.functional as F


class VAnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(VAnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc_A=torch.nn.Linear(hidden_dim,action_dim)
        self.fc_V=torch.nn.Linear(hidden_dim,1)
    
    
    def forward(self,x):
        A=self.fc_A(F.relu(self.fc1(x)))
        V=self.fc_V(F.relu(self.fc1(x)))
        Q=V+A-A.mean(1).view(-1,1)
        return Q
    
    
class DuelingDQN:
    def __init__(self,state_dim,hidden_dim,action_dim):
        if torch.cuda.is_available():
            self.device_d=torch.device('cuda')
            self.device_n=torch.device('cuda')
        else:
            self.device_d=torch.device('cpu')
            self.device_n=torch.device('cpu')
        self.nn=VAnet(state_dim,hidden_dim,action_dim).to(self.device_n)
        self.target_q_net=VAnet(state_dim,hidden_dim,action_dim).to(self.device_n)
        self.optimizer=torch.optim.Adam(self.nn.parameters(),lr=2e-3)
        self.genv=gym.make('CartPole-v0')
    
    
    def env(self,a=None,initial=None):
        if initial==True:
            state=self.genv.reset(seed=0)
            return state
        else:
            next_state,reward,done,_=self.genv.step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d):
        s=torch.tensor(s,dtype=torch.float).to(self.device_d)
        a=torch.tensor(a,dtype=torch.int64).view(-1,1).to(self.device_d)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device_d)
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(self.device_d)
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(self.device_d)
        q_value=self.nn(s).gather(1,a)
        next_q_value=self.target_q_net(next_s).max(1)[0].view(-1,1)
        target=r+0.98*next_q_value*(1-d)
        return F.mse_loss(q_value,target)
    
    
    def backward(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        return
    
    
    def opt(self):
        self.optimizer.step()
        return
        
    
    def update_param(self):
        self.target_q_net.load_state_dict(self.nn.state_dict())
        return
