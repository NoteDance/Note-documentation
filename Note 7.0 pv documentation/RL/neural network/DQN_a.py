import torch
import gym
import torch.nn.functional as F
import Note.create.RL.rl.assign_a as assign_a

#gradient attenuation example
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
            self.device_d=torch.device('cuda')
            self.device_n=torch.device('cuda')
        else:
            self.device_d=torch.device('cpu')
            self.device_n=torch.device('cpu')
        self.nn=Qnet(state_dim,hidden_dim,action_dim).to(self.device_n)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim).to(self.device_n)
        self.optimizer=torch.optim.Adam(self.nn.parameters(),lr=2e-3)
        self.genv=gym.make('CartPole-v0')
        self.oc={}
        self.grad={}

    
    
    def env(self,a=None,initial=None):
        if initial==True:
            self.genv.action_space.seed(0)
            state,info=self.genv.reset(seed=0,return_info=True)
            return state
        else:
            next_state,reward,done,_=self.genv.step(a)
            if done:
                next_state,info=self.genv.reset(return_info=True)
            return next_state,reward,done
    
    
    def loss(self,s,a,next_s,r,d):
        s=torch.tensor(s,dtype=torch.float).to(self.device_d)
        a=torch.tensor(a).view(-1,1).to(self.device_d)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device_d)
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(self.device_d)
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(self.device_d)
        q_value=self.nn(s).gather(1,a)
        next_q_value=self.target_q_net(next_s).max(1)[0].view(-1,1)
        target=r+0.98*next_q_value*(1-d)
        return F.mse_loss(q_value,target)
    
    
    def attenuate(self,model,oc,grad):
        #complete attenuation function
        assign_a.assign(model,ac,grad)
        
    
    def backward(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        return
    
    
    def opt(self,t):
        self.attenuate(self.nn,self.oc[t],self.grad[t])
        self.optimizer.step()
        return
        
    
    def update_param(self):
        self.target_q_net.load_state_dict(self.nn.state_dict())
        return
