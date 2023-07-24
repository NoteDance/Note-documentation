import torch # import the PyTorch library
from torch import nn # import the neural network module from PyTorch
from Note.nn.assign_device import assign_device_pytorch # import a custom function to assign device for PyTorch tensors


class NeuralNetwork(nn.Module): # define a class for the neural network model
    def __init__(self): # define the constructor method
        super().__init__() # call the parent class constructor
        self.flatten = nn.Flatten() # define a layer to flatten the input tensor
        self.linear_relu_stack=nn.Sequential( # define a stack of linear and relu layers
            nn.Linear(28*28,512), # define a linear layer with input size 28*28 and output size 512
            nn.ReLU(), # define a relu activation function
            nn.Linear(512,512), # define another linear layer with input and output size 512
            nn.ReLU(), # define another relu activation function
            nn.Linear(512,10) # define the final linear layer with output size 10
        )
    
    
    def forward(self,x): # define the forward method for the model
        x = self.flatten(x) # flatten the input tensor
        logits=self.linear_relu_stack(x) # pass the flattened tensor through the linear relu stack
        return logits # return the logits as the output


class neuralnetwork: # define another class for the neural network object
    def __init__(self): # define the constructor method
        if torch.cuda.is_available(): # check if cuda device is available
            self.device=torch.device('cuda') # set the device to cuda
        else: # otherwise
            self.device=torch.device('cpu') # set the device to cpu
        self.model=NeuralNetwork().to(self.device) # create an instance of the neural network model and move it to the device
        self.loss_fn=nn.CrossEntropyLoss() # define the loss function as cross entropy loss
        self.opt=[torch.optim.SGD(self.model.parameters(),lr=1e-3) for _ in range(7)] # define a list of optimizers as stochastic gradient descent with learning rate 1e-3
    
    
    def fp(self,x,p): # define a method for forward propagation
        pred=self.model(x.to(assign_device_pytorch(p,'GPU')))   # assign the device according to the process index p
                                                                # get the predictions from the model by passing the input tensor to the device and then to the model
        return pred # return the predictions
    
    
    def loss(self,output,labels,p): # define a method for calculating loss
        loss=self.loss_fn(output,labels.to(assign_device_pytorch(p,'GPU'))) # assign the device according to the process index p
                                                                            # calculate the loss by passing the output and labels tensors to the device and then to the loss function
        return loss # return the loss
