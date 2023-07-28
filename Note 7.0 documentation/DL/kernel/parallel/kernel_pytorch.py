import torch # import the torch library
from multiprocessing import Value,Array # import the Value and Array classes from the multiprocessing module
import numpy as np # import the numpy library


class kernel: # define a class named kernel
    def __init__(self,nn=None): # define the constructor method of the class
        self.nn=nn # assign the nn argument to the self.nn attribute
        if hasattr(self.nn,'km'): # check if the nn attribute has an attribute named km
            self.nn.km=1 # set the km attribute of the nn attribute to 1
        self.process=None # initialize the process attribute to None
        self.process_t=None # initialize the process_t attribute to None
        self.train_ds=None # initialize the train_ds attribute to None
        self.batches_t=None # initialize the batches_t attribute to None
        self.shuffle=False # initialize the shuffle attribute to False
        self.priority_flag=False # initialize the priority_flag attribute to False
        self.max_opt=None # initialize the max_opt attribute to None
        self.epoch=None # initialize the epoch attribute to None
        self.stop=False # initialize the stop attribute to False
        self.save_epoch=None # initialize the save_epoch attribute to None
        self.batch=None # initialize the batch attribute to None
        self.end_loss=None # initialize the end_loss attribute to None
        self.end_acc=None # initialize the end_acc attribute to None
        self.end_test_loss=None # initialize the end_test_loss attribute to None
        self.end_test_acc=None # initialize the end_test_acc attribute to None
        self.acc_flag='%' # initialize the acc_flag attribute to '%'
        self.s=None # initialize the s attribute to None
        self.saving_one=True # initialize the saving_one attribute to True
        self.filename='save.dat' # initialize the filename attribute to 'save.dat'
        self.test_flag=False # initialize the test_flag attribute to False
    
    
    def data(self,train_dataset=None,test_dataset=None): 
        '''define a method named data that takes two arguments: train_dataset and test_dataset'''
        
        '''This method is used to assign and process the train_dataset and test_dataset attributes, 
        and also create some arrays for storing some statistics'''
        
        '''The train_dataset and test_dataset arguments are expected to be PyTorch Dataset objects or lists of PyTorch Dataset objects'''
        
        '''The train_dataset argument is used for training the neural network'''
        
        '''The test_dataset argument is used for testing or evaluating  the neural network, and it is optional'''
        
        '''If the test_dataset argument is given, the test_flag attribute will be set to True, 
        indicating that the test method will be called after each epoch'''
    
        self.train_dataset=train_dataset # assign the train_dataset argument to the self.train_dataset attribute
        self.test_dataset=test_dataset # assign the test_dataset argument to the self.test_dataset attribute
        if test_dataset is not None: # check if the test_dataset argument is not None
            self.test_flag=True # set the test_flag attribute to True
        self.batch_counter=np.zeros(self.process,dtype=np.int32) # create an array of zeros with a length equal to the process attribute and an integer data type, and assign it to the batch_counter attribute
        self.total_loss=np.zeros(self.process) # create an array of zeros with a length equal to the process attribute and a float data type, and assign it to the total_loss attribute
        if hasattr(self.nn,'accuracy'): # check if the nn attribute has a method named accuracy
            if type(self.nn.param[0])!=list: # check if the first element of the param attribute of the nn attribute is not a list
                self.total_acc=np.zeros(self.process,dtype=self.nn.param[0].dtype.name) # create an array of zeros with a length equal to the process attribute and a data type equal to the data type of the first element of the param attribute of the nn attribute, and assign it to the total_acc attribute
            else: # otherwise
                self.total_acc=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name) # create an array of zeros with a length equal to the process attribute and a data type equal to the data type of the first element of the first element of the param attribute of the nn attribute, and assign it to the total_acc attribute
        if self.priority_flag==True: # check if the priority_flag attribute is True
            self.opt_counter=np.zeros(self.process,dtype=np.int32) # create an array of zeros with a length equal to the process attribute and an integer data type, and assign it to the opt_counter attribute
        return # return nothing
    
    
    def init(self,manager): 
        '''define a method named init that takes one argument: manager'''
        
        '''This method is used to initialize some shared variables for multiprocessing, 
        using the manager argument as a multiprocessing.Manager object'''
        
        '''The manager argument is expected to be a multiprocessing.Manager object that can create shared variables across processes'''
    
        self.epoch_counter=Value('i',0) # create a shared variable with an integer data type and an initial value equal to the epoch_counter attribute, and assign it to the epoch_counter attribute
        self.batch_counter=Array('i',self.batch_counter) # create a shared array with an integer data type and an initial value equal to the batch_counter attribute, and assign it to the batch_counter attribute
        self.total_loss=Array('f',self.total_loss) # create a shared array with a float data type and an initial value equal to the total_loss attribute, and assign it to the total_loss attribute
        self.total_epoch=Value('i',0) # create a shared variable with an integer data type and an initial value equal to the total_epoch attribute, and assign it to the total_epoch attribute
        self.train_loss=Value('f',0) # create a shared variable with a float data type and an initial value equal to the train_loss attribute, and assign it to the train_loss attribute
        self.train_loss_list=manager.list([]) # create a shared list with an initial value equal to the train_loss_list attribute, using the list method of the manager argument, and assign it to the train_loss_list attribute
        self.priority_p=Value('i',0) # create a shared variable with an integer data type and an initial value equal to the priority_p attribute, and assign it to the priority_p attribute
        if self.test_flag==True: # check if the test_flag attribute is True
            self.test_loss=Value('f',0) # create a shared variable with a float data type and an initial value equal to the test_loss attribute, and assign it to the test_loss attribute
            self.test_loss_list=manager.list([]) # create a shared list with an initial value equal to the test_loss_list attribute, using the list method of the manager argument, and assign it to the test_loss_list attribute
        if hasattr(self.nn,'accuracy'): # check if the nn attribute has a method named accuracy
            self.total_acc=Array('f',self.total_acc) # create a shared array with a float data type and an initial value equal to the total_acc attribute, and assign it to the total_acc attribute
            self.train_acc=Value('f',0) # create a shared variable with a float data type and an initial value equal to the train_acc attribute, and assign it to the train_acc attribute
            self.train_acc_list=manager.list([]) # create a shared list with an initial value equal to the train_acc_list attribute, using the list method of the manager argument, and assign it to the train_acc_list attribute
            if self.test_flag==True: # check if the test_flag attribute is True
                self.test_acc=Value('f',0) # create a shared variable with a float data type and an initial value equal to the test_acc attribute, and assign it to the test_acc attribute
                self.test_acc_list=manager.list([]) # create a shared list with an initial value equal to the test_acc_list attribute, using the list method of the manager argument, and assign it to the test_acc_list attribute
        if self.priority_flag==True: # check if the priority_flag attribute is True
            self.opt_counter=Array('i',self.opt_counter) # create a shared array with an integer data type and an initial value equal to the opt_counter attribute, and assign it to the opt_counter attribute  
        try:
            self.nn.opt_counter=manager.list([self.nn.opt_counter]) # try to create a shared list with an initial value equal to a list containing the opt_counter attribute of the nn attribute, using the list method of the manager argument, and assign it to the opt_counter attribute of the nn attribute  
        except Exception: # handle any exception that may occur
            self.opt_counter_=manager.list() # create an empty shared list using the list method of the manager argument, and assign it to the opt_counter_ attribute
        try:
            self.nn.ec=manager.list([self.nn.ec]) # try to create a shared list with an initial value equal to a list containing the ec attribute of the nn attribute, using the list method of the manager argument, and assign it to the ec attribute of the nn attribute  
        except Exception: # handle any exception that may occur
            self.ec_=manager.list() # create an empty shared list using the list method of the manager argument, and assign it to the ec_ attribute
        try:
            self.nn.bc=manager.list([self.nn.bc]) # try to create a shared list with an initial value equal to a list containing the bc attribute of the nn attribute, using the list method of the manager argument, and assign it to the bc attribute of the nn attribute
        except Exception: # handle any exception that may occur
            self.bc_=manager.list() # create an empty shared list using the list method of the manager argument, and assign it to the bc_ attribute
        self.epoch_=Value('i',0) # create a shared variable with an integer data type and an initial value equal to the epoch_ attribute, and assign it to the epoch_ attribute
        self.stop_flag=Value('b',False) # create a shared variable with a boolean data type and an initial value equal to the stop_flag attribute, and assign it to the stop_flag attribute
        self.save_flag=Value('b',False) # create a shared variable with a boolean data type and an initial value equal to the save_flag attribute, and assign it to the save_flag attribute
        self.file_list=manager.list([]) # create an empty shared list using the list method of the manager argument, and assign it to the file_list attribute
        return # return nothing
    
    
    def init_online(self,manager): 
        '''define a method named init_online that takes one argument: manager'''
        
        '''This method is used to initialize some shared variables for online training, 
        using the manager argument as a multiprocessing.Manager object'''
        
        '''The manager argument is expected to be a multiprocessing.Manager object that can create shared variables across processes'''
        
        self.nn.train_loss_list=manager.list([]) # create an empty shared list using the list method of the manager argument, and assign it to the train_loss_list attribute of the nn attribute
        self.nn.train_acc_list=manager.list([]) # create an empty shared list using the list method of the manager argument, and assign it to the train_acc_list attribute of the nn attribute
        self.nn.counter=manager.list([]) # create an empty shared list using the list method of the manager argument, and assign it to the counter attribute of the nn attribute
        self.nn.exception_list=manager.list([]) # create an empty shared list using the list method of the manager argument, and assign it to the exception_list attribute of the nn attribute
        return # return nothing
    
    
    def end(self): 
        '''define a method named end that takes no arguments'''
        
        '''This method is used to check if the training process should be stopped based on some criteria'''
        
        '''The criteria are based on some attributes of the class, such as end_loss, end_acc, end_test_loss, and end_test_acc'''
        
        '''These attributes are expected to be numbers that represent some thresholds for training loss, training accuracy, test loss, and test accuracy respectively'''
        
        '''The method will return a boolean value indicating whether the training process should be stopped or not'''
        
        if self.end_loss!=None and len(self.train_loss_list)!=0 and self.train_loss_list[-1]<self.end_loss: # check if the end_loss attribute is not None and the train_loss_list attribute is not empty and the last element of the train_loss_list attribute is less than the end_loss attribute
            return True # return True
        elif self.end_acc!=None and len(self.train_acc_list)!=0 and self.train_acc_list[-1]>self.end_acc: # check if the end_acc attribute is not None and the train_acc_list attribute is not empty and the last element of the train_acc_list attribute is greater than the end_acc attribute
            return True # return True
        elif self.end_loss!=None and len(self.train_loss_list)!=0 and self.end_acc!=None and self.train_loss_list[-1]<self.end_loss and self.train_acc_list[-1]>self.end_acc: # check if both previous conditions are met
            return True # return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.test_loss_list[-1]<self.end_test_loss: # check if the end_test_loss attribute is not None and the test_loss_list attribute is not empty and the last element of the test_loss_list attribute is less than the end_test_loss attribute
            return True # return True
        elif self.end_test_acc!=None and len(self.test_acc_list)!=0 and self.test_acc_list[-1]>self.end_test_acc: # check if the end_test_acc attribute is not None and the test_acc_list attribute is not empty and the last element of the test_acc_list attribute is greater than the end_test_acc attribute
            return True # return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.end_test_acc!=None and self.test_loss_list[-1]<self.end_test_loss and self.test_acc_list[-1]>self.end_test_acc: # check if both previous conditions are met
            return True # return True
        else: # otherwise
            return False # return False
    
    
    def opt_p(self,data,labels,p): 
        '''define a method named opt_p that takes three arguments: data, labels, and p'''
        
        '''This method is used to perform one step of optimization for the neural network, 
        using the data and labels arguments as the input and output of the neural network, 
        and the p argument as the index of the process'''
        
        '''The data and labels arguments are expected to be PyTorch tensors that represent a batch of input and output data for the neural network'''
        
        '''The p argument is expected to be an integer that represents the index of the process that calls this method'''
        
        try: # try to execute the following code block
            try: # try to execute the following code block
                try: # try to execute the following code block
                    output=self.nn.fp(data,p) # call the fp method of the nn attribute with the data and p arguments, and assign the return value to the output variable
                    loss=self.nn.loss(output,labels,p) # call the loss method of the nn attribute with the output, labels, and p arguments, and assign the return value to the loss variable
                except Exception: # handle any exception that may occur
                    output,loss=self.nn.fp(data,labels,p) # call the fp method of the nn attribute with the data, labels, and p arguments, and assign the return values to the output and loss variables
            except Exception: # handle any exception that may occur
                try: # try to execute the following code block
                    output=self.nn.fp(data) # call the fp method of the nn attribute with the data argument, and assign the return value to the output variable
                    loss=self.nn.loss(output,labels) # call the loss method of the nn attribute with the output and labels arguments, and assign the return value to the loss variable
                except Exception: # handle any exception that may occur
                    output,loss=self.nn.fp(data,labels) # call the fp method of the nn attribute with the data and labels arguments, and assign the return values to the output and loss variables
        except Exception as e: # handle any exception that may occur
            raise e # raise the exception again
        if self.priority_flag==True and self.priority_p.value!=-1: # check if both priority_flag attribute is True and priority_p attribute is not -1
            while True: # enter an infinite loop
                if p==self.priority_p.value: # check if p is equal to priority_p attribute
                    break # break out of the loop
                else: # otherwise
                    continue # continue looping
        self.nn.opt[p].zero_grad(set_to_none=True) # call the zero_grad method of the opt attribute of the nn attribute with the p argument, and set the set_to_none argument to True
        loss=loss.clone() # create a copy of the loss variable and assign it to the loss variable
        loss.backward() # call the backward method of the loss variable to compute the gradients
        self.nn.opt[p].step() # call the step method of the opt attribute of the nn attribute with the p argument to update the parameters
        return output,loss # return the output and loss variables
    
    
    def opt(self,data,labels,p): 
        '''define a method named opt that takes three arguments: data, labels, and p'''
        
        '''This method is used to perform one step of optimization for the neural network, 
        using the data and labels arguments as the input and output of the neural network, 
        and the p argument as the index of the process'''
        
        '''The data and labels arguments are expected to be PyTorch tensors that represent a batch of input and output data for the neural network'''
        
        '''The p argument is expected to be an integer that represents the index of the process that calls this method'''
        
        output,loss=self.opt_p(data,labels,p) # call the opt_p method with the data, labels, and p arguments, and assign the return values to the output and loss variables
        return output,loss # return the output and loss variables
    
    
    def train7(self,train_loader,p,test_batch,lock): 
        '''define a method named train7 that takes four arguments: train_loader, p, test_batch, and lock'''
        
        '''This method is used to train the neural network for one epoch using a given train_loader, 
        using the p argument as the index of the process, 
        using the test_batch argument as a batch of test data for evaluation, 
        and using the lock argument as a multiprocessing.Lock object for synchronization'''
        
        '''The train_loader argument is expected to be a PyTorch DataLoader object that provides batches of training data for the neural network'''
        
        '''The p argument is expected to be an integer that represents the index of the process that calls this method'''
        
        '''The test_batch argument is expected to be a PyTorch tensor that represents a batch of test data for evaluation'''
        
        '''The lock argument is expected to be a multiprocessing.Lock object that can be used to acquire and release a lock for synchronization across processes'''
        while True: # enter an infinite loop
            for data_batch,labels_batch in train_loader: # iterate over the batches of training data from the train_loader
                if hasattr(self.nn,'data_func'): # check if the nn attribute has a method named data_func
                    data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch) # call the data_func method of the nn attribute with the data_batch and labels_batch arguments, and assign the return values to the data_batch and labels_batch variables
                if self.priority_flag==True: # check if the priority_flag attribute is True
                    self.priority_p.value=np.argmax(self.opt_counter) # set the value of the priority_p attribute to the index of the maximum element of the opt_counter attribute
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt: # check if both max_opt attribute is not None and the element of the opt_counter attribute at the index of the priority_p attribute is greater than or equal to the max_opt attribute
                        self.priority_p.value=int(self.priority_p.value) # cast the value of the priority_p attribute to an integer
                    elif self.max_opt==None: # check if max_opt attribute is None
                        self.priority_p.value=int(self.priority_p.value) # cast the value of the priority_p attribute to an integer
                    else: # otherwise
                        self.priority_p.value=-1 # set the value of the priority_p attribute to -1
                if self.priority_flag==True: # check if the priority_flag attribute is True
                    self.opt_counter[p]=0 # set the element of the opt_counter attribute at the index of p to 0
                if hasattr(self.nn,'attenuate'): # check if the nn attribute has a method named attenuate
                    opt_counter=self.nn.opt_counter[0] # assign the first element of the opt_counter attribute of the nn attribute to the opt_counter variable
                    opt_counter[p]=0 # set the element of the opt_counter variable at the index of p to 0
                    self.nn.opt_counter[0]=opt_counter # assign the opt_counter variable to the first element of the opt_counter attribute of the nn attribute
                output,batch_loss=self.opt(data_batch,labels_batch,p) # call the opt method with the data_batch, labels_batch, and p arguments, and assign the return values to the output and batch_loss variables
                if self.priority_flag==True: # check if the priority_flag attribute is True
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i') # create a numpy array from a buffer object that contains a copy of memory from another object, using an integer data type, and assign it to the opt_counter variable
                    opt_counter+=1 # increment each element of the opt_counter variable by 1
                if hasattr(self.nn,'attenuate'): # check if the nn attribute has a method named attenuate
                    opt_counter=self.nn.opt_counter[0] # assign the first element of the opt_counter attribute of the nn attribute to the opt_counter variable
                    opt_counter+=1 # increment each element of the opt_counter variable by 1
                    self.nn.opt_counter[0]=opt_counter # assign the opt_counter variable to the first element of the opt_counter attribute of the nn attribute
                if hasattr(self.nn,'bc'): # check if the nn attribute has an attribute named bc
                    bc=self.nn.bc[0] # assign the first element of the bc attribute of the nn attribute to the bc variable
                    bc+=1 # increment the bc variable by 1
                    self.nn.bc[0]=bc # assign the bc variable to the first element of the bc attribute of the nn attribute
                try: # try to execute the following code block
                    if hasattr(self.nn,'accuracy'): # check if the nn attribute has a method named accuracy
                        try: # try to execute the following code block
                            batch_acc=self.nn.accuracy(output,labels_batch,p) # call the accuracy method of the nn attribute with the output, labels_batch, and p arguments, and assign the return value to the batch_acc variable
                        except Exception: # handle any exception that may occur
                            batch_acc=self.nn.accuracy(output,labels_batch) # call the accuracy method of the nn attribute with the output and labels_batch arguments, and assign the return value to the batch_acc variable
                except Exception as e: # handle any exception that may occur
                    raise e # raise the exception again
                if hasattr(self.nn,'accuracy'): # check if the nn attribute has a method named accuracy
                    self.total_loss[p]+=batch_loss # increment the element of the total_loss attribute at the index of p by the batch_loss variable
                    self.total_acc[p]+=batch_acc # increment the element of the total_acc attribute at the index of p by the batch_acc variable
                else: # otherwise
                    self.total_loss[p]+=batch_loss # increment the element of the total_loss attribute at the index of p by the batch_loss variable
                self.batch_counter[p]+=1 # increment the element of the batch_counter attribute at the index of p by 1
                if lock is not None: # check if lock argument is not None
                    lock.acquire() # call the acquire method of lock argument to acquire a lock for synchronization
                batches=np.sum(self.batch_counter) # compute the sum of all elements in the batch_counter attribute and assign it to batches variable
                if batches>=len(train_loader): # check if batches is greater than or equal to length of train_loader argument
                    batch_counter=np.frombuffer(self.batch_counter.get_obj(),dtype='i') # create a numpy array from a buffer object that contains a copy of memory from another object, using an integer data type, and assign it to batch_counter variable
                    batch_counter*=0 # multiply each element of batch_counter variable by 0
                    loss=np.sum(self.total_loss)/batches # compute the average of all elements in total_loss attribute and assign it to loss variable
                    if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
                        train_acc=np.sum(self.total_acc)/batches # compute the average of all elements in total_acc attribute and assign it to train_acc variable
                    self.total_epoch.value+=1 # increment value of total_epoch attribute by 1
                    self.train_loss.value=loss # assign loss variable to value of train_loss attribute
                    self.train_loss_list.append(loss) # append loss variable to train_loss_list attribute
                    if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
                        self.train_acc.value=train_acc # assign train_acc variable to value of train_acc attribute
                        self.train_acc_list.append(train_acc) # append train_acc variable to train_acc_list attribute
                    if self.test_flag==True: # check if test_flag attribute is True
                        if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
                            self.test_loss.value,self.test_acc.value=self.test(test_batch) # call the test method with the test_batch argument, and assign the return values to the value of the test_loss attribute and the value of the test_acc attribute
                            self.test_loss_list.append(self.test_loss.value) # append the value of the test_loss attribute to the test_loss_list attribute
                            self.test_acc_list.append(self.test_acc.value) # append the value of the test_acc attribute to the test_acc_list attribute
                        else: # otherwise
                            self.test_loss.value=self.test(test_batch) # call the test method with the test_batch argument, and assign the return value to the value of the test_loss attribute
                            self.test_loss_list.append(self.test_loss.value) # append the value of the test_loss attribute to the test_loss_list attribute
                self.save_() # call the save_ method to save
                self.epoch_counter.value+=1 # increment the value of the epoch_counter attribute by 1
                if hasattr(self.nn,'ec'): # check if the nn attribute has an attribute named ec
                    ec=self.nn.ec[0] # assign the first element of the ec attribute of the nn attribute to the ec variable
                    ec+=1 # increment the ec variable by 1
                    self.nn.ec[0]=ec # assign the ec variable to the first element of the ec attribute of the nn attribute
                total_loss=np.frombuffer(self.total_loss.get_obj(),dtype='f') # create a numpy array from a buffer object that contains a copy of memory from another object, using a float data type, and assign it to total_loss variable
                total_loss*=0 # multiply each element of total_loss variable by 0
                if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
                    total_acc=np.frombuffer(self.total_acc.get_obj(),dtype='f') # create a numpy array from a buffer object that contains a copy of memory from another object, using a float data type, and assign it to total_acc variable
                    total_acc*=0 # multiply each element of total_acc variable by 0
                if lock is not None: # check if lock argument is not None
                    lock.release() # call the release method of lock argument to release a lock for synchronization
                if self.epoch_counter.value>=self.epoch: # check if value of epoch_counter attribute is greater than or equal to epoch attribute
                    return # return nothing
    
    
    def train(self,p,lock=None,test_batch=None): 
        '''define a method named train that takes three arguments: p, lock, and test_batch'''
        
        '''This method is used to train the neural network for one epoch using a given train_dataset, 
        using the p argument as the index of the process, 
        using the lock argument as a multiprocessing.Lock object for synchronization, 
        and using the test_batch argument as a batch of test data for evaluation'''
        
        '''The p argument is expected to be an integer that represents the index of the process that calls this method'''
        
        '''The lock argument is expected to be a multiprocessing.Lock object that can be used to acquire and release a lock for synchronization across processes'''
        
        '''The test_batch argument is expected to be a PyTorch tensor that represents a batch of test data for evaluation'''
        if type(self.train_dataset)==list: # check if the train_dataset attribute is a list
            train_loader=torch.utils.data.DataLoader(self.train_dataset[p],batch_size=self.batch) # create a PyTorch DataLoader object from the element of the train_dataset attribute at the index of p, using the batch attribute as the batch size, and assign it to the train_loader variable
        else: # otherwise
            train_loader=torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch,shuffle=self.shuffle) # create a PyTorch DataLoader object from the train_dataset attribute, using the batch attribute as the batch size and the shuffle attribute as the shuffle flag, and assign it to the train_loader variable
        self.train7(train_loader,p,test_batch,lock) # call the train7 method with the train_loader, p, test_batch, and lock arguments
        return # return nothing
    
    
    def train_online(self,p,lock=None): 
        '''define a method named train_online that takes two arguments: p and lock'''
        
        '''This method is used to train the neural network online using a given online method of the neural network, 
        using the p argument as the index of the process, 
        and using the lock argument as a multiprocessing.Lock object for synchronization'''
        
        '''The p argument is expected to be an integer that represents the index of the process that calls this method'''
        
        '''The lock argument is expected to be a multiprocessing.Lock object that can be used to acquire and release a lock for synchronization across processes'''
        
        if hasattr(self.nn,'counter'): # check if the nn attribute has an attribute named counter
            self.nn.counter.append(0) # append 0 to the counter attribute of the nn attribute
        while True: # enter an infinite loop
            if hasattr(self.nn,'save'): # check if the nn attribute has a method named save
                self.nn.save(self.save,p) # call the save method of the nn attribute with the save and p arguments
            if hasattr(self.nn,'stop_flag'): # check if the nn attribute has an attribute named stop_flag
                if self.nn.stop_flag==True: # check if the value of the stop_flag attribute of the nn attribute is True
                    return # return nothing
            if hasattr(self.nn,'stop_func'): # check if the nn attribute has a method named stop_func
                if self.nn.stop_func(p): # call the stop_func method of the nn attribute with the p argument, and check if it returns True
                    return # return nothing
            if hasattr(self.nn,'suspend_func'): # check if the nn attribute has a method named suspend_func
                self.nn.suspend_func(p) # call the suspend_func method of the nn attribute with the p argument
            try: # try to execute the following code block
                data=self.nn.online(p) # call the online method of the nn attribute with the p argument, and assign the return value to data variable
            except Exception as e: # handle any exception that may occur
                self.nn.exception_list[p]=e # assign e to the element of the exception_list attribute of the nn attribute at the index of p
            if data=='stop': # check if data is equal to 'stop'
                return # return nothing
            elif data=='suspend': # check if data is equal to 'suspend'
                self.nn.suspend_func(p) # call the suspend_func method of the nn attribute with the p argument
            try: # try to execute the following code block
                output,loss,param=self.opt(data[0],data[1],p,lock) # call the opt method with the first and second elements of data, p, and lock arguments, and assign the return values to output, loss, and param variables
                self.param[7]=param # assign param variable to the seventh element of the param attribute
            except Exception as e: # handle any exception that may occur
                if lock!=None: # check if lock argument is not None
                    if lock.acquire(False): # call the acquire method of lock argument with False argument, and check if it returns True
                        lock.release() # call the release method of lock argument to release a lock for synchronization
                self.nn.exception_list[p]=e # assign e to the element of the exception_list attribute of the nn attribute at the index of p
            loss=loss.numpy() # convert loss variable to a numpy array and assign it to loss variable
            if len(self.nn.train_loss_list)==self.nn.max_length: # check if length of train_loss_list attribute of nn attribute is equal to max_length attribute of nn attribute
                del self.nn.train_loss_list[0] # delete the first element of train_loss_list attribute of nn attribute
            self.nn.train_loss_list.append(loss) # append loss variable to train_loss_list attribute of nn attribute
            try: # try to execute the following code block
                if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
                    try: # try to execute the following code block
                        acc=self.nn.accuracy(output,data[1]) # call the accuracy method of nn attribute with output and second element of data arguments, and assign the return value to acc variable
                    except Exception: # handle any exception that may occur
                        self.exception_list[p]=True # assign True to the element of exception_list attribute at the index of p
                    if len(self.nn.train_acc_list)==self.nn.max_length: # check if length of train_acc_list attribute of nn attribute is equal to max_length attribute of nn attribute
                        del self.nn.train_acc_list[0] # delete the first element of train_acc_list attribute of nn attribute
                    self.nn.train_acc_list.append(acc) # append acc variable to train_acc_list attribute of nn attribute
            except Exception as e: # handle any exception that may occur
                self.nn.exception_list[p]=e # assign e to the element of exception_list attribute of nn attribute at the index of p
            try: # try to execute the following code block
                if hasattr(self.nn,'counter'): # check if nn attribute has an attribute named counter
                    count=self.nn.counter[p] # assign the element of counter attribute of nn attribute at the index of p to count variable
                    count+=1 # increment count variable by 1
                    self.nn.counter[p]=count # assign count variable to the element of counter attribute of nn attribute at the index of p
            except IndexError: # handle any IndexError that may occur
                self.nn.counter.append(0) # append 0 to counter attribute of nn attribute
                count=self.nn.counter[p] # assign the element of counter attribute of nn attribute at the index of p to count variable
                count+=1 # increment count variable by 1
                self.nn.counter[p]=count # assign count variable to the element of counter attribute of nn attribute at the index of p
        return # return nothing
    
    
    def test_(self,data,labels): 
        '''define a method named test_ that takes two arguments: data and labels'''
        
        '''This method is used to compute the test loss and test accuracy for a given batch of test data, 
        using the data and labels arguments as the input and output of the neural network'''
        
        '''The data and labels arguments are expected to be PyTorch tensors that represent a batch of input and output data for the neural network'''
        
        try: # try to execute the following code block
            try: # try to execute the following code block
                output=self.nn.fp(data) # call the fp method of nn attribute with data argument, and assign the return value to output variable
                loss=self.nn.loss(output,labels) # call the loss method of nn attribute with output and labels arguments, and assign the return value to loss variable
            except Exception: # handle any exception that may occur
                output,loss=self.nn.fp(data,labels) # call the fp method of nn attribute with data and labels arguments, and assign the return values to output and loss variables
        except Exception as e: # handle any exception that may occur
            raise e # raise the exception again
        try: # try to execute the following code block
            acc=self.nn.accuracy(output,labels) # call the accuracy method of nn attribute with output and labels arguments, and assign the return value to acc variable
        except Exception as e: # handle any exception that may occur
            if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
                raise e # raise the exception again
            else: # otherwise
                acc=None # assign None to acc variable
        return loss,acc # return loss and acc variables
    
    
    def test(self,batch=None): 
        '''define a method named test that takes one argument: batch'''
        
        '''This method is used to compute the test loss and test accuracy for the whole test dataset or a given batch of test data, 
        using the batch argument as a batch of test data for evaluation'''
        
        '''The batch argument is expected to be a PyTorch tensor that represents a batch of test data for evaluation, or None to use the whole test dataset'''
        total_loss=0 # initialize total_loss variable to 0
        total_acc=0 # initialize total_acc variable to 0
        batches=0 # initialize batches variable to 0
        if batch is None: # check if batch argument is None
            test_loader=torch.utils.data.DataLoader(self.test_dataset,batch_size=self.batch) # create a PyTorch DataLoader object from the test_dataset attribute, using the batch attribute as the batch size, and assign it to the test_loader variable
        else: # otherwise
            test_loader=[batch] # create a list containing the batch argument and assign it to the test_loader variable
        for data_batch,labels_batch in test_loader: # iterate over the batches of test data from the test_loader
            batches+=1 # increment batches variable by 1
            batch_loss,batch_acc=self.test_(data_batch,labels_batch) # call the test_ method with the data_batch and labels_batch arguments, and assign the return values to batch_loss and batch_acc variables
            total_loss+=batch_loss # increment total_loss variable by batch_loss variable
            if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
                total_acc+=batch_acc # increment total_acc variable by batch_acc variable
        test_loss=total_loss.detach().numpy()/batches # compute the average of total_loss variable and convert it to a numpy array and assign it to test_loss variable
        if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
            test_acc=total_acc.detach().numpy()/batches # compute the average of total_acc variable and convert it to a numpy array and assign it to test_acc variable
        if hasattr(self.nn,'accuracy'): # check if nn attribute has a method named accuracy
            return test_loss,test_acc # return test_loss and test_acc variables
        else: # otherwise
            return test_loss # return test_loss variable
