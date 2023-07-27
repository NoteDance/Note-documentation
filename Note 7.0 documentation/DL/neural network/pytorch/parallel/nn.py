import torch # import the PyTorch library
from torch import nn # import the neural network module from PyTorch

class NeuralNetwork(nn.Module): # define a class for the neural network model
    def __init__(self): # define the constructor method
        super().__init__() # call the parent class constructor
        self.flatten = nn.Flatten() # define a layer to flatten the input
        self.linear_relu_stack=nn.Sequential( # define a stack of linear and relu layers
            nn.Linear(28*28,512), # a linear layer with 28*28 input features and 512 output features
            nn.ReLU(), # a relu activation function
            nn.Linear(512,512), # another linear layer with 512 input and output features
            nn.ReLU(), # another relu activation function
            nn.Linear(512,10) # a final linear layer with 512 input features and 10 output features
        )
    
    
    def forward(self,x): # define the forward method
        x = self.flatten(x) # flatten the input x
        logits=self.linear_relu_stack(x) # pass x through the linear relu stack and get the logits
        return logits # return the logits


class neuralnetwork: # define another class for the neural network object
    def __init__(self): # define the constructor method
        if torch.cuda.is_available(): # check if cuda is available
            self.device=torch.device('cuda') # set the device to cuda
        else:
            self.device=torch.device('cpu') # otherwise set the device to cpu
        self.model=NeuralNetwork().to(self.device) # create an instance of the NeuralNetwork class and move it to the device
        self.loss_fn=nn.CrossEntropyLoss() # define the loss function as cross entropy loss
        self.opt=[torch.optim.SGD(self.model.parameters(),lr=1e-3) for _ in range(7)] # define a list of three stochastic gradient descent optimizers with learning rate 1e-3
    
    
    def fp(self,x): # define a method for forward propagation
        pred=self.model(x.to(self.device)) # get the predictions from the model by passing x to the device
        return pred # return the predictions
    
    
    def loss(self,output,labels): # define a method for calculating the loss
        loss=self.loss_fn(output,labels.to(self.device)) # compute the loss by passing output and labels to the device
        return loss # return the loss