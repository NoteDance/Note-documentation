import tensorflow as tf
import Note.nn.layer.dense as d
import Note.nn.process.optimizer as o
from Note.nn.layer.flatten import flatten

# Define a neural network class with gradient attenuation
class nn:
    def __init__(self):
        # Initialize the loss function and the optimizer
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=o.Momentum(0.07,0.7)
        # Initialize a variable to keep track of the number of optimization steps
        self.opt_counter=tf.Variable(tf.zeros(7,dtype=tf.float32))
    
    
    def build(self):
        # Create two dense layers with relu and linear activations
        self.layer1=d.dense([784,128],activation='relu')
        self.layer2=d.dense([128,10])
        # Store the parameters of the layers in a list
        self.param=[self.layer1.param_list,self.layer2.param_list]
        return
    
    
    def fp(self,data):
        # Perform forward propagation on the input data
        data=flatten(data)
        output1=self.layer1.output(data)
        output2=self.layer2.output(output1)
        return output2
    
    
    def loss(self,output,labels):
        # Compute the loss between the output and the labels
        return self.loss_object(labels,output)
    
    
    def attenuate(self,gradient,oc,p):  #gradient attenuation function,kernel uses it to calculate attenuation coefficient.
        # Apply an exponential decay to the gradient based on the optimization counter
        ac=0.9**oc[p]                   #ac:attenuation coefficient
        for i in range(len(gradient)):  #oc:optimization counter
            gradient[i]=ac*gradient[i]  #p:process number
        return gradient  
    

    def opt(self,gradient):
        # Perform optimization on the parameters using the gradient
        param=self.optimizer.opt(gradient,self.param)
        return param                            
