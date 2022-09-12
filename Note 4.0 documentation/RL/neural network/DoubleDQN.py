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
    
    
class DoubleDQN:
    def __init__(self,state_dim,hidden_dim,action_dim):
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.nn=Qnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.target_q_net(state_dim,hidden_dim,action_dim).to(self.device)
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=2e-3)
        self.env=gym.make('CartPole-v0')

    
    
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
    
    
    def loss(self,s,a,next_s,r,d):
        s=torch.tensor(s,dtype=torch.float).to(self.device)
        a=torch.tensor(a,dtype=torch.float).view(-1,1).to(self.device)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device)
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(self.device)
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(self.device)
        q_value=self.nn(s).gather(1,a)
        max_action=self.nn(next_s).max(1)[1].view(-1,1)
        next_q_value=self.target_q_net(next_s).gather(1,max_action)
        target=r+0.98*next_q_value*(1-d)
        return torch.mean(F.mse_loss(q_value,target))
    
    
    def opt(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def update_param(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())