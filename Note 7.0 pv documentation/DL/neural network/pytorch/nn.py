import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits


class neuralnetwork:
    def __init__(self):
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.model=NeuralNetwork().to(self.device)
        self.loss_fn=nn.CrossEntropyLoss()
        self.opt=torch.optim.SGD(self.model.parameters(),lr=1e-3) #optimizer,kernel uses this to optimize.
    
    
    def fp(self,x):   #forward propagation function,kernel uses this for forward propagation.
        pred=self.model(x.to(self.device))
        return pred
    
    
    def loss(self,output,labels):  #loss functino,kernel uses this to calculate loss.
        loss=self.loss_fn(output,labels.to(self.device))
        return loss
    
    
    def backward(self,loss): #backward function,kernel uses this for backpropagation.
        self.optim.zero_grad()
        loss.backward()
        return
    
    
    def opt(self): #opt function,kernel uses this to optimize.
        self.optim.step()
        return
