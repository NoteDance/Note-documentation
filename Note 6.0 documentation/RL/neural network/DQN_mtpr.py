import torch
import gym
import torch.nn.functional as F
import Note.create.RL.rl.prioritized_replay as pr

#mutithreading prioritized replay example
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
        self.pr=pr.pr_mt()
        self.initial_TD=7
        self._epsilon=0.0007
        self.alpha=0.7
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=2e-3)
        self.genv=gym.make('CartPole-v0')

    
    
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
    
    
    def data_func(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,batch,t):
        s,a,next_s,r,d=self.pr.sample(state_pool,action_pool,next_state_pool,reward_pool,done_pool,self.epsilon,self.alpha,batch,t)
        return s,a,next_s,r,d
        
    
    def loss(self,s,a,next_s,r,d,t):
        s=torch.tensor(s,dtype=torch.float).to(self.device)
        a=torch.tensor(a).view(-1,1).to(self.device)
        next_s=torch.tensor(next_s,dtype=torch.float).to(self.device)
        r=torch.tensor(r,dtype=torch.float).view(-1,1).to(self.device)
        d=torch.tensor(d,dtype=torch.float).view(-1,1).to(self.device)
        q_value=self.nn(s).gather(1,a)
        next_q_value=self.target_q_net(next_s).max(1)[0].view(-1,1)
        target=r+0.98*next_q_value*(1-d)
        TD=target-q_value
        self.pr.update_TD(TD,t)
        return torch.mean(TD**2)
    
    
    def opt(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return
        
    
    def update_param(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        return
