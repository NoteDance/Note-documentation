import torch
import gym
import torch.nn.functional as F
from Note.nn.process.assign_device_pytorch import assign_device


class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)
    
    
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim):
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.nn=Qnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim).to(self.device)
        self.optimizer=[torch.optim.Adam(self.nn.parameters(),lr=2e-3) for _ in range(5)] #optimizer,kernel uses it to optimize.
        self.genv=[gym.make('CartPole-v0') for _ in range(5)] #create environment
    
    
    def env(self,a=None,p=None,initial=None): #environment function,kernel uses it to interact with the environment.
        if initial==True:
            state=self.genv[p].reset()
            return state
        else:
            next_state,reward,done,_=self.genv[p].step(a)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d,p): #loss function,kernel uses it to calculate loss.
        s=torch.tensor(s,dtype=torch.float).to(assign_device(p,'GPU'))
        a=torch.tensor(a,dtype=torch.int64).view(-1,1).to(assign_device(p,'GPU'))
        next_s=torch.tensor(next_s,dtype=torch.float).to(assign_device(p,'GPU'))
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(assign_device(p,'GPU'))
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(assign_device(p,'GPU'))
        q_value=self.nn(s).gather(1,a)
        next_q_value=self.target_q_net(next_s).max(1)[0].view(-1,1)
        target=r+0.98*next_q_value*(1-d)
        return F.mse_loss(q_value,target)
    
    
    def backward(self,loss,p): #backward function,kernel uses it for backpropagation.
        self.optimizer[p].zero_grad(set_to_none=True)
        loss.backward()
        return
    
    
    def opt(self,p): #opt function,kernel uses it to optimize.
        self.optimizer[p].step()
        return
        
    
    def update_param(self): #update function,kernel uses it to update parameter.
        self.target_q_net.load_state_dict(self.nn.state_dict())
        return
