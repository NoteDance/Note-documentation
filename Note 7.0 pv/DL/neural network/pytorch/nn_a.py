import torch
from torch import nn
import Note.create.DL.dl.assign_a as assign_a

#gradient attenuation example
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits


class nn:
    def __init__(self,device):
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.model=NeuralNetwork().to(self.device)
        self.loss_fn=nn.CrossEntropyLoss()
        self.optim=torch.optim.SGD(self.model.parameters(),lr=1e-3)
    
    
    def fp(self,x):
        pred=self.model(x.to(self.device))
        return pred
    
    
    def loss(self,output,labels):
        loss=self.loss_fn(output,labels.to(self.device))
        return loss
    
    
    def attenuate(self,oc):
        #complete attenuation function
        assign_a(self.model,oc)
    
    
    def opt(self,loss,oc):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return
