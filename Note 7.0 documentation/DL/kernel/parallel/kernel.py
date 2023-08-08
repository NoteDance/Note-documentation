import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Process,Value,Array
import numpy as np
from Note.DL.dl.test import parallel_test


class kernel:
    def __init__(self,nn=None):
        self.nn=nn # the neural network model
        if hasattr(self.nn,'km'):
            self.nn.km=1 # a flag to indicate the kernel mode
        self.PO=None # the optimization strategy
        self.process=None # the number of processes for training
        self.process_t=None # the number of processes for testing
        self.train_ds=None # the training dataset
        self.prefetch_batch_size=tf.data.AUTOTUNE # the prefetch batch size for training dataset
        self.prefetch_batch_size_t=tf.data.AUTOTUNE # the prefetch batch size for testing dataset
        self.data_segment_flag=False # a flag to indicate whether to segment the data for each process
        self.batches=None # the number of batches per epoch
        self.buffer_size=None # the buffer size for shuffling the data
        self.priority_flag=False # a flag to indicate whether to use priority optimization
        self.max_opt=None # the maximum number of optimization steps for each process
        self.epoch=None # the number of epochs for training
        self.stop=False # a flag to indicate whether to stop the training
        self.save_epoch=None # the epoch interval for saving the model parameters
        self.batch=None # the batch size for training and testing data
        self.end_loss=None # the end condition for training loss
        self.end_acc=None # the end condition for training accuracy
        self.end_test_loss=None # the end condition for testing loss
        self.end_test_acc=None # the end condition for testing accuracy
        self.acc_flag='%' # a flag to indicate whether to use percentage or decimal for accuracy display
        self.s=None # initialize the s attribute to None
        self.saving_one=True # a flag to indicate whether to save only one copy of model parameters or multiple copies with different names
        self.filename='save.dat' # the default filename for saving model parameters
        self.test_flag=False # a flag to indicate whether to use testing data or not
        
        
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        # a method to set the data for training and testing
        if train_data is not None and type(self.nn.param[0])!=list:
            self.train_data=train_data.astype(self.nn.param[0].dtype.name) # convert the training data to the same dtype as the model parameters
            self.train_labels=train_labels.astype(self.nn.param[0].dtype.name) # convert the training labels to the same dtype as the model parameters
        elif train_data is not None:
            self.train_data=train_data.astype(self.nn.param[0][0].dtype.name) # convert the training data to the same dtype as the model parameters
            self.train_labels=train_labels.astype(self.nn.param[0][0].dtype.name) # convert the training labels to the same dtype as the model parameters
        self.train_dataset=train_dataset # set the training dataset
        self.test_data=test_data # set the testing data
        self.test_labels=test_labels # set the testing labels
        self.test_dataset=test_dataset # set the testing dataset
        if test_data is not None or test_dataset is not None:
            self.test_flag=True # set the test flag to True if there is testing data or dataset
        self.batch_counter=np.zeros(self.process,dtype=np.int32) # initialize a counter for batches for each process
        if type(self.nn.param[0])!=list:
            self.total_loss=np.zeros(self.process,dtype=self.nn.param[0].dtype.name) # initialize a total loss for each process with the same dtype as the model parameters
        else:
            self.total_loss=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name) # initialize a total loss for each process with the same dtype as the model parameters
        if hasattr(self.nn,'accuracy'):
            if type(self.nn.param[0])!=list:
                self.total_acc=np.zeros(self.process,dtype=self.nn.param[0].dtype.name) # initialize a total accuracy for each process with the same dtype as the model parameters
            else:
                self.total_acc=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name) # initialize a total accuracy for each process with the same dtype as the model parameters
        if self.priority_flag==True:
            self.opt_counter=np.zeros(self.process,dtype=np.int32) # initialize a counter for optimization steps for each process if using priority optimization
        if train_data is not None:
            self.shape0=train_data.shape[0] # get the number of samples in training data
            self.batches=int((self.shape0-self.shape0%self.batch)/self.batch) # calculate the number of batches per epoch
            if self.shape0%self.batch!=0:
                self.batches+=1 # add one more batch if there are some remaining samples
        if self.data_segment_flag==True:
            self.train_data,self.train_labels=self.segment_data() # segment the data for each process if using data segment flag
        return
    
    
    def segment_data(self):
        # a method to segment the data for each process
        # calculate the number of data to be sliced
        length=len(self.train_data)-len(self.train_data)%self.process
        # slice the front part of the data and labels
        data=self.train_data[:length]
        labels=self.train_labels[:length]
        # split the data and labels into subarrays
        data=np.split(data, self.process)
        labels=np.split(labels, self.process)
        # stack the data and labels along a new axis
        data=np.stack(data, axis=0)
        labels=np.stack(labels, axis=0)
        return data,labels
    
    
    def init(self,manager):
        # a method to initialize some shared variables for multiprocessing
        self.epoch_counter=Value('i',0) # create a shared value for epoch counter
        self.batch_counter=Array('i',self.batch_counter) # create a shared array for batch counter
        self.total_loss=Array('f',self.total_loss) # create a shared array for total loss
        self.total_epoch=Value('i',0) # create a shared value for total epoch
        self.train_loss=Value('f',0) # create a shared value for training loss
        self.train_loss_list=manager.list([]) # create a shared list for training loss list
        self.priority_p=Value('i',0) # create a shared value for priority process index
        if self.test_flag==True: 
            self.test_loss=Value('f',0) # create a shared value for testing loss if using testing data or dataset
            self.test_loss_list=manager.list([]) # create a shared list for testing loss list if using testing data or dataset
        if hasattr(self.nn,'accuracy'):
            if self.nn.accuracy!=None:
                self.total_acc=Array('f',self.total_acc) # create a shared array for total accuracy if using accuracy metric
                self.train_acc=Value('f',0) # create a shared value for training accuracy if using accuracy metric
                self.train_acc_list=manager.list([]) # create a shared list for training accuracy list if using accuracy metric
                if self.test_flag==True:
                    self.test_acc=Value('f',0) # create a shared value for testing accuracy if using testing data or dataset and accuracy metric
                    self.test_acc_list=manager.list([]) # create a shared list for testing accuracy list if using testing data or dataset and accuracy metric
        if self.priority_flag==True:
            self.opt_counter=Array('i',self.opt_counter)  # create a shared array for optimization counter if using priority optimization 
        try:
            self.nn.opt_counter=manager.list([self.nn.opt_counter])  # create a shared list for optimization counter in the neural network model 
        except Exception:
            self.opt_counter_=manager.list() # create an empty list if there is no optimization counter in the neural network model 
        try:
            self.nn.ec=manager.list([self.nn.ec])  # create a shared list for epoch counter in the neural network model 
        except Exception:
            self.ec_=manager.list() # create an empty list if there is no epoch counter in the neural network model 
        try:
            self.nn.bc=manager.list([self.nn.bc]) # create a shared list for batch counter in the neural network model 
        except Exception:
            self.bc_=manager.list() # create an empty list if there is no batch counter in the neural network model 
        self.epoch_=Value('i',0) # create a shared value for epoch counter in online mode
        self.stop_flag=Value('b',False) # create a shared value for stop flag by external signal
        self.save_flag=Value('b',False) # create a shared value for save flag by external signal
        self.file_list=manager.list([]) # create an empty list for file names of saved model parameters
        self.param=manager.dict() # create an empty dictionary for model parameters
        self.param[7]=self.nn.param # set the 7th key of the dictionary to be the model parameters of the neural network model 
        return
    
    
    def init_online(self,manager):
        # a method to initialize some shared variables for online learning
        self.nn.train_loss_list=manager.list([]) # create an empty list for training loss list in online mode in the neural network model 
        self.nn.train_acc_list=manager.list([]) # create an empty list for training accuracy list in online mode in the neural network model 
        self.nn.counter=manager.list([]) # create an empty list for counter in online mode in the neural network model 
        self.nn.exception_list=manager.list([]) # create an empty list for exception list in online mode in the neural network model 
        self.param=manager.dict() # create an empty dictionary for model parameters
        self.param[7]=self.nn.param # set the 7th key of the dictionary to be the model parameters of the neural network model 
        return
    
    
    def end(self):
        # a method to check whether to end the training according to some conditions
        if self.end_loss!=None and len(self.train_loss_list)!=0 and self.train_loss_list[-1]<self.end_loss:
            return True # return True if the training loss is lower than the end loss
        elif self.end_acc!=None and len(self.train_acc_list)!=0 and self.train_acc_list[-1]>self.end_acc:
            return True # return True if the training accuracy is higher than the end accuracy
        elif self.end_loss!=None and len(self.train_loss_list)!=0 and self.end_acc!=None and self.train_loss_list[-1]<self.end_loss and self.train_acc_list[-1]>self.end_acc:
            return True # return True if both the training loss and accuracy meet the end conditions
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.test_loss_list[-1]<self.end_test_loss:
            return True # return True if the testing loss is lower than the end test loss
        elif self.end_test_acc!=None and len(self.test_acc_list)!=0 and self.test_acc_list[-1]>self.end_test_acc:
            return True # return True if the testing accuracy is higher than the end test accuracy
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.end_test_acc!=None and self.test_loss_list[-1]<self.end_test_loss and self.test_acc_list[-1]>self.end_test_acc:
            return True # return True if both the testing loss and accuracy meet the end conditions
    
    
    @tf.function(jit_compile=True)
    def opt_p(self,data,labels,p,lock,g_lock=None):
        # a method to perform one optimization step for a given process
        try:
            try:
                with tf.GradientTape(persistent=True) as tape: # create a gradient tape to record the gradients
                    try:
                        try:
                            output=self.nn.fp(data,p) # get the output of the neural network model with data and process index as inputs
                            loss=self.nn.loss(output,labels,p) # get the loss of the output with labels and process index as inputs
                        except Exception:
                            output,loss=self.nn.fp(data,labels,p) # get the output and loss of the neural network model with data, labels and process index as inputs
                    except Exception:
                        try:
                            output=self.nn.fp(data) # get the output of the neural network model with data as input
                            loss=self.nn.loss(output,labels) # get the loss of the output with labels as input
                        except Exception:
                            output,loss=self.nn.fp(data,labels) # get the output and loss of the neural network model with data and labels as inputs
            except Exception:
                if hasattr(self.nn,'GradientTape'):
                    tape,output,loss=self.nn.GradientTape(data,labels,p) # use a custom gradient tape method in the neural network model with data, labels and process index as inputs
        except Exception as e:
            raise e # raise any exception that occurs
        if self.PO==1: # if using PO=1 strategy (lock before calculating gradients)
            if self.priority_flag==True and self.priority_p.value!=-1: # if using priority optimization and there is a priority process index
                while True: 
                    if p==self.priority_p.value: # wait until this process is equal to the priority process index
                        break 
                    else: 
                        continue 
            lock[0].acquire() # acquire the first lock (for calculating gradients)
            if self.stop_func_(lock[0]): # check whether to stop the training by external signal
                return None,0 # return None values if stopping
            try:
                if hasattr(self.nn,'gradient'): # if the neural network model has a custom gradient method
                    try:
                        gradient=self.nn.gradient(tape,loss) # get the gradients with the tape and loss as inputs
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7]) # get the gradients with the tape, loss and model parameters as inputs
                else:
                    gradient=tape.gradient(loss,self.nn.param) # get the gradients with the tape and model parameters as inputs
            except Exception as e:
                raise e # raise any exception that occurs
            if hasattr(self.nn,'attenuate'): # if the neural network model has a custom attenuate method for modifying the gradients
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # modify the gradients with the optimization counter and process index as inputs
            try:
                try:
                    param=self.nn.opt(gradient) # get the updated model parameters with the gradients as input
                except Exception:
                    param=self.nn.opt(gradient,p) # get the updated model parameters with the gradients and process index as inputs
            except Exception as e:
                raise e # raise any exception that occurs
            lock[0].release() # release the first lock (for calculating gradients)
        elif self.PO==2: # if using PO=2 strategy (lock before and after calculating gradients)
            g_lock.acquire() # acquire the global lock (for calculating gradients)
            if self.stop_func_(g_lock): # check whether to stop the training by external signal
                return None,0 # return None values if stopping
            try:
                if hasattr(self.nn,'gradient'): # if the neural network model has a custom gradient method
                    try:
                        gradient=self.nn.gradient(tape,loss) # get the gradients with the tape and loss as inputs
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7]) # get the gradients with the tape, loss and model parameters as inputs
                else:
                    gradient=tape.gradient(loss,self.nn.param) # get the gradients with the tape and model parameters as inputs
            except Exception as e:
                raise e # raise any exception that occurs
            g_lock.release() # release the global lock (for calculating gradients)
            if self.priority_flag==True and self.priority_p.value!=-1: # if using priority optimization and there is a priority process index
                while True: 
                    if p==self.priority_p.value: # wait until this process is equal to the priority process index
                        break 
                    else: 
                        continue 
            lock[0].acquire() # acquire the first lock (for updating model parameters)
            if self.stop_func_(lock[0]): # check whether to stop the training by external signal
                return None,0 # return None values if stopping
            if hasattr(self.nn,'attenuate'): # if the neural network model has a custom attenuate method for modifying the gradients
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # modify the gradients with the optimization counter and process index as inputs
            try:
                try:
                    param=self.nn.opt(gradient) # get the updated model parameters with the gradients as input
                except Exception:
                    param=self.nn.opt(gradient,p) # get the updated model parameters with the gradients and process index as inputs
            except Exception as e:
                raise e # raise any exception that occurs
            lock[0].release() # release the first lock (for updating model parameters)
        elif self.PO==3: # if using PO=3 strategy (no lock for calculating gradients)
            if self.priority_flag==True and self.priority_p.value!=-1: # if using priority optimization and there is a priority process index
                while True: 
                    if p==self.priority_p.value: # wait until this process is equal to the priority process index
                        break 
                    else: 
                        continue 
            if self.stop_func_(): # check whether to stop the training by external signal
                return None,0 # return None values if stopping
            try:
                if hasattr(self.nn,'gradient'): # if the neural network model has a custom gradient method
                    try:
                        gradient=self.nn.gradient(tape,loss) # get the gradients with the tape and loss as inputs
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7]) # get the gradients with the tape, loss and model parameters as inputs
                else:
                    gradient=tape.gradient(loss,self.nn.param) # get the gradients with the tape and model parameters as inputs
            except Exception as e:
                raise e # raise any exception that occurs
            if hasattr(self.nn,'attenuate'): # if the neural network model has a custom attenuate method for modifying the gradients
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # modify the gradients with the optimization counter and process index as inputs
            try:
                try:
                    param=self.nn.opt(gradient) # get the updated model parameters with the gradients as input
                except Exception:
                    param=self.nn.opt(gradient,p) # get the updated model parameters with the gradients and process index as inputs
            except Exception as e:
                raise e # raise any exception that occurs
        return output,loss,param # return the output, loss and updated model parameters
    
    
    def opt(self,data,labels,p,lock,g_lock):
        # a method to perform one optimization step for a given process with different PO strategies
        if self.PO==2: # if using PO=2 strategy (lock before and after calculating gradients)
            if type(g_lock)!=list: 
                pass # do nothing if g_lock is not a list
            elif len(g_lock)==self.process: 
                ln=p # set ln to be the process index
                g_lock=g_lock[ln] # get the global lock for this process
            else: 
                ln=int(np.random.choice(len(g_lock))) # randomly choose a global lock from the list
                g_lock=g_lock[ln] # get the global lock for this process
            output,loss,param=self.opt_p(data,labels,p,lock,g_lock) # perform one optimization step with data, labels, process index, lock and global lock as inputs
        else: 
            output,loss,param=self.opt_p(data,labels,p,lock) # perform one optimization step with data, labels, process index and lock as inputs
        return output,loss,param # return the output, loss and updated model parameters
    
    
    def update_nn_param(self,param=None):
        # a method to update the model parameters of the neural network model with the shared dictionary values or a given parameter value
        if param==None: 
            parameter_flat=nest.flatten(self.nn.param) # flatten the model parameters of the neural network model into a list
            parameter7_flat=nest.flatten(self.param[7]) # flatten the 7th value of the shared dictionary into a list
        else: 
            parameter_flat=nest.flatten(self.nn.param) # flatten the model parameters of the neural network model into a list
            parameter7_flat=nest.flatten(param) # flatten the given parameter value into a list
        for i in range(len(parameter_flat)): 
            if param==None: 
                state_ops.assign(parameter_flat[i],parameter7_flat[i]) # assign each element of parameter7_flat to each element of parameter_flat
            else: 
                state_ops.assign(parameter_flat[i],parameter7_flat[i]) # assign each element of parameter7_flat to each element of parameter_flat
        self.nn.param=nest.pack_sequence_as(self.nn.param,parameter_flat) # pack the parameter_flat list back into the original structure of self.nn.param
        self.param[7]=nest.pack_sequence_as(self.param[7],parameter7_flat) # pack the parameter7_flat list back into the original structure of self.param[7]
        return
    
    
    def train7(self,train_ds,p,test_batch,lock,g_lock):
        # a method to perform one epoch of training for a given process with a given training dataset
        while True: 
            for data_batch,labels_batch in train_ds: # iterate over each batch of data and labels in the training dataset
                if hasattr(self.nn,'data_func'): 
                    data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch) # apply a custom data function in the neural network model to the data and labels batch
                if self.priority_flag==True: 
                    self.priority_p.value=np.argmax(self.opt_counter) # set the priority process index to be the one with the maximum optimization counter
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt: 
                        self.priority_p.value=int(self.priority_p.value) # keep the priority process index as an integer if it reaches the maximum optimization steps
                    elif self.max_opt==None: 
                        self.priority_p.value=int(self.priority_p.value) # keep the priority process index as an integer if there is no maximum optimization steps
                    else: 
                        self.priority_p.value=-1 # set the priority process index to -1 if there is no process with maximum optimization steps
                if self.priority_flag==True: 
                    self.opt_counter[p]=0 # reset the optimization counter for this process if using priority optimization
                if hasattr(self.nn,'attenuate'): 
                    opt_counter=self.nn.opt_counter[0] # get the optimization counter in the neural network model
                    opt_counter.scatter_update(tf.IndexedSlices(0,p)) # update the optimization counter for this process to 0
                    self.nn.opt_counter[0]=opt_counter # set the optimization counter in the neural network model
                output,batch_loss,param=self.opt(data_batch,labels_batch,p,lock,g_lock) # perform one optimization step with data and labels batch, process index, lock and global lock as inputs
                self.param[7]=param # set the 7th value of the shared dictionary to be the updated model parameters
                if self.priority_flag==True: 
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i') # get the optimization counter from the shared array
                    opt_counter+=1 # increment the optimization counter by 1 for each process
                if hasattr(self.nn,'attenuate'): 
                    opt_counter=self.nn.opt_counter[0] # get the optimization counter in the neural network model
                    opt_counter.assign(opt_counter+1) # increment the optimization counter by 1 for this process
                    self.nn.opt_counter[0]=opt_counter # set the optimization counter in the neural network model
                if hasattr(self.nn,'bc'): 
                    bc=self.nn.bc[0] # get the batch counter in the neural network model
                    bc.assign_add(1) # increment the batch counter by 1 for this process
                    self.nn.bc[0]=bc # set the batch counter in the neural network model
                try:
                    if hasattr(self.nn,'accuracy'): 
                        try:
                            batch_acc=self.nn.accuracy(output,labels_batch,p) # get the accuracy of the output with labels and process index as inputs
                        except Exception:
                            batch_acc=self.nn.accuracy(output,labels_batch) # get the accuracy of the output with labels as inputs
                except Exception as e:
                    raise e # raise any exception that occurs
                if hasattr(self.nn,'accuracy'): 
                    self.total_loss[p]+=batch_loss # accumulate the total loss for this process
                    self.total_acc[p]+=batch_acc # accumulate the total accuracy for this process
                else: 
                    self.total_loss[p]+=batch_loss # accumulate the total loss for this process
                self.batch_counter[p]+=1 # increment the batch counter for this process
                if self.PO==1 or self.PO==2: 
                    lock[1].acquire() # acquire the second lock (for printing and saving)
                elif lock!=None: 
                    lock.acquire() # acquire a single lock (for printing and saving)
                batches=np.sum(self.batch_counter) # get the total number of batches for all processes
                if batches>=self.batches: # check whether all batches are finished for this epoch
                    batch_counter=np.frombuffer(self.batch_counter.get_obj(),dtype='i') # get the batch counter from the shared array
                    batch_counter*=0 # reset the batch counter to 0 for each process
                    loss=np.sum(self.total_loss)/batches # calculate the average training loss for this epoch
                    if hasattr(self.nn,'accuracy'): 
                        train_acc=np.sum(self.total_acc)/batches # calculate the average training accuracy for this epoch
                    self.total_epoch.value+=1 # increment the total epoch by 1
                    self.train_loss.value=loss # set the training loss value to be the average loss
                    self.train_loss_list.append(loss) # append the average loss to the training loss list
                    if hasattr(self.nn,'accuracy'): 
                        self.train_acc.value=train_acc # set the training accuracy value to be the average accuracy
                        self.train_acc_list.append(train_acc) # append the average accuracy to the training accuracy list
                    if self.test_flag==True: # if using testing data or dataset
                        if hasattr(self.nn,'accuracy'): 
                            self.test_loss.value,self.test_acc.value=self.test(self.test_data,self.test_labels,test_batch) # get the testing loss and accuracy values with testing data, labels and batch size as inputs
                            self.test_loss_list.append(self.test_loss.value) # append the testing loss value to the testing loss list
                            self.test_acc_list.append(self.test_acc.value) # append the testing accuracy value to the testing accuracy list
                        else: 
                            self.test_loss.value=self.test(self.test_data,self.test_labels,test_batch) # get the testing loss value with testing data, labels and batch size as inputs
                            self.test_loss_list.append(self.test_loss.value) # append the testing loss value to the testing loss list
                    self.save_() # call the save_ method to save
                    self.epoch_counter.value+=1 # increment the epoch counter by 1
                    if hasattr(self.nn,'ec'): 
                        ec=self.nn.ec[0] # get the epoch counter in the neural network model
                        ec.assign_add(1) # increment the epoch counter by 1 for this process
                        self.nn.ec[0]=ec # set the epoch counter in the neural network model
                    total_loss=np.frombuffer(self.total_loss.get_obj(),dtype='f') # get the total loss from the shared array
                    total_loss*=0 # reset the total loss to 0 for each process
                    if hasattr(self.nn,'accuracy'): 
                        total_acc=np.frombuffer(self.total_acc.get_obj(),dtype='f') # get the total accuracy from the shared array
                        total_acc*=0 # reset the total accuracy to 0 for each process
                if self.PO==1 or self.PO==2: 
                    lock[1].release() # release the second lock (for printing and saving)
                elif lock!=None: 
                    lock.release() # release a single lock (for printing and saving)
                if self.epoch_counter.value>=self.epoch: 
                    self.param[7]=param # set the 7th value of the shared dictionary to be the updated model parameters
                    return # return from this method
    
    
    def train(self,p,lock=None,g_lock=None,test_batch=None):
        # a method to perform one epoch of training for a given process with different PO strategies and data sources
        if self.train_dataset is not None and type(self.train_dataset)==list: 
            train_ds=self.train_dataset[p] # get the training dataset for this process if it is a list of datasets
        elif self.train_dataset is not None: 
            train_ds=self.train_dataset # get the training dataset if it is a single dataset
        else: 
            if self.data_segment_flag==True: 
                train_ds=tf.data.Dataset.from_tensor_slices((self.train_data[p],self.train_labels[p])).batch(self.batch).prefetch(self.prefetch_batch_size) # create a tf.data.Dataset object from segmented data and labels for this process with batch size and prefetch size as inputs
            elif self.buffer_size!=None: 
                train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).shuffle(self.buffer_size).batch(self.batch).prefetch(self.prefetch_batch_size) # create a tf.data.Dataset object from shuffled data and labels with buffer size, batch size and prefetch size as inputs
            else: 
                train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).batch(self.batch).prefetch(self.prefetch_batch_size) # create a tf.data.Dataset object from data and labels with batch size and prefetch size as inputs
        self.train7(train_ds,p,test_batch,lock,g_lock) # perform one epoch of training for this process with the training dataset, process index, test batch size, lock and global lock as inputs
        return
    
    
    def train_online(self,p,lock=None,g_lock=None):
        # a method to perform online training for a given process with different PO strategies
        if hasattr(self.nn,'counter'): 
            self.nn.counter.append(0) # append a 0 to the counter list in the neural network model for this process
        while True: 
            if hasattr(self.nn,'save'): 
                self.nn.save(self.save,p) # save the model parameters with the save path and process index as inputs if the neural network model has a save method
            if hasattr(self.nn,'stop_flag'): 
                if self.nn.stop_flag==True: # check whether to stop the training by the stop flag in the neural network model
                    return # return from this method
            if hasattr(self.nn,'stop_func'): 
                if self.nn.stop_func(p): # check whether to stop the training by the stop function in the neural network model with process index as input
                    return # return from this method
            if hasattr(self.nn,'suspend_func'): 
                self.nn.suspend_func(p) # suspend the training by the suspend function in the neural network model with process index as input
            try:
                data=self.nn.online(p) # get the online data from the online method in the neural network model with process index as input
            except Exception as e:
                self.nn.exception_list[p]=e # append any exception that occurs to the exception list in the neural network model for this process
            if data=='stop': 
                return # return from this method if the online data is 'stop'
            elif data=='suspend': 
                self.nn.suspend_func(p) # suspend the training by the suspend function in the neural network model with process index as input if the online data is 'suspend'
            try:
                if self.PO==2: # if using PO=2 strategy (lock before and after calculating gradients)
                    if type(g_lock)!=list: 
                        pass # do nothing if g_lock is not a list
                    elif len(g_lock)==self.process: 
                        ln=p # set ln to be the process index
                        g_lock=g_lock[ln] # get the global lock for this process
                else: 
                    ln=int(np.random.choice(len(g_lock))) # randomly choose a global lock from the list
                    g_lock=g_lock[ln] # get the global lock for this process
                output,loss,param=self.opt(data[0],data[1],p,lock,g_lock) # perform one optimization step with data and labels, process index, lock and global lock as inputs
                self.param[7]=param # set the 7th value of the shared dictionary to be the updated model parameters
            except Exception as e:
                if self.PO==1: 
                    if lock[0].acquire(False): 
                        lock[0].release() # release the first lock if it is acquired by this process
                elif self.PO==2: 
                    if g_lock.acquire(False): 
                        g_lock.release() # release the global lock if it is acquired by this process
                    if lock[0].acquire(False): 
                        lock[0].release() # release the first lock if it is acquired by this process
                self.nn.exception_list[p]=e # append any exception that occurs to the exception list in the neural network model for this process
            loss=loss.numpy() # convert the loss value to a numpy array
            if len(self.nn.train_loss_list)==self.nn.max_length: 
                del self.nn.train_loss_list[0] # delete the first element of the training loss list in the neural network model if it reaches the maximum length
            self.nn.train_loss_list.append(loss) # append the loss value to the training loss list in the neural network model
            try:
                if hasattr(self.nn,'accuracy'): 
                    try:
                        acc=self.nn.accuracy(output,data[1]) # get the accuracy of the output with labels as inputs
                    except Exception:
                        self.exception_list[p]=True # set the exception flag to True for this process if there is an exception
                    if len(self.nn.train_acc_list)==self.nn.max_length: 
                        del self.nn.train_acc_list[0] # delete the first element of the training accuracy list in the neural network model if it reaches the maximum length
                    self.nn.train_acc_list.append(acc) # append the accuracy value to the training accuracy list in the neural network model
            except Exception as e:
                self.nn.exception_list[p]=e # append any exception that occurs to the exception list in the neural network model for this process
            try:
                if hasattr(self.nn,'counter'): 
                    count=self.nn.counter[p] # get the counter value for this process in the neural network model
                    count+=1 # increment the counter value by 1
                    self.nn.counter[p]=count # set the counter value for this process in the neural network model
            except IndexError: 
                self.nn.counter.append(0) # append a 0 to the counter list in the neural network model for this process if there is an index error
                count=self.nn.counter[p] # get the counter value for this process in the neural network model
                count+=1 # increment the counter value by 1
                self.nn.counter[p]=count # set the counter value for this process in the neural network model
        return # return from this method
    
    
    @tf.function(jit_compile=True)
    def test_(self,data,labels):
        # a method to perform one testing step with data and labels as inputs
        try:
            try:
                output=self.nn.fp(data) # get the output of the neural network model with data as input
                loss=self.nn.loss(output,labels) # get the loss of the output with labels as input
            except Exception:
                output,loss=self.nn.fp(data,labels) # get the output and loss of the neural network model with data and labels as inputs
        except Exception as e:
            raise e # raise any exception that occurs
        try:
            acc=self.nn.accuracy(output,labels) # get the accuracy of the output with labels as input
        except Exception as e:
            if hasattr(self.nn,'accuracy'): 
                raise e # raise any exception that occurs if using accuracy metric
            else: 
                acc=None # set acc to None if not using accuracy metric
        return loss,acc # return the loss and accuracy values
    
    
    def test(self,test_data=None,test_labels=None,batch=None):
        # a method to perform testing with different data sources and batch sizes
        if test_data is not None and type(self.nn.param[0])!=list: 
            test_data=test_data.astype(self.nn.param[0].dtype.name) # convert the testing data to the same dtype as the model parameters
            test_labels=test_labels.astype(self.nn.param[0].dtype.name) # convert the testing labels to the same dtype as the model parameters
        elif test_data is not None: 
            test_data=test_data.astype(self.nn.param[0][0].dtype.name) # convert the testing data to the same dtype as the model parameters
            test_labels=test_labels.astype(self.nn.param[0][0].dtype.name) # convert the testing labels to the same dtype as the model parameters
        if self.process_t!=None: # if using multiple processes for testing
            parallel_test_=parallel_test(self.nn,self.test_data,self.test_labels,self.process_t,batch,self.prefetch_batch_size_t,self.test_dataset) # create a parallel_test object with the neural network model, testing data, labels, number of processes, batch size, prefetch size and dataset as inputs
            if type(self.test_data)!=list: 
                parallel_test_.segment_data() # segment the data for each process if it is not a list of data
            for p in range(self.process_t): 
                Process(target=parallel_test_.test,args=(p,)).start() # start a subprocess to perform testing for each process
            try:
                if hasattr(self.nn,'accuracy'): 
                    test_loss,test_acc=parallel_test_.loss_acc() # get the testing loss and accuracy values from the parallel_test object
            except Exception as e:
                if hasattr(self.nn,'accuracy'): 
                    raise e # raise any exception that occurs if using accuracy metric
                else: 
                    test_loss=parallel_test_.loss_acc() # get the testing loss value from the parallel_test object
        elif batch!=None: # if using a batch size for testing
            total_loss=0 # initialize a total loss value
            total_acc=0 # initialize a total accuracy value
            if self.test_dataset!=None: # if using a testing dataset
                batches=0 # initialize a batches counter
                for data_batch,labels_batch in self.test_dataset: # iterate over each batch of data and labels in the testing dataset
                    batches+=1 # increment the batches counter by 1
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch) # perform one testing step with data and labels batch as inputs
                    total_loss+=batch_loss # accumulate the total loss value
                    if hasattr(self.nn,'accuracy'): 
                        total_acc+=batch_acc # accumulate the total accuracy value if using accuracy metric
                test_loss=total_loss.numpy()/batches # calculate the average testing loss value
                if hasattr(self.nn,'accuracy'): 
                    test_acc=total_acc.numpy()/batches # calculate the average testing accuracy value if using accuracy metric
            else: 
                shape0=test_data.shape[0] # get the number of samples in testing data
                batches=int((shape0-shape0%batch)/batch) # calculate the number of batches for testing data
                for j in range(batches): 
                    index1=j*batch # get the start index of each batch
                    index2=(j+1)*batch # get the end index of each batch
                    data_batch=test_data[index1:index2] # get a slice of testing data for each batch
                    labels_batch=test_labels[index1:index2] # get a slice of testing labels for each batch
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch) # perform one testing step with data and labels batch as inputs
                    total_loss+=batch_loss # accumulate the total loss value
                    if hasattr(self.nn,'accuracy'): 
                        total_acc+=batch_acc # accumulate the total accuracy value if using accuracy metric
                if shape0%batch!=0: # if there are some remaining samples in testing data
                    batches+=1 # add one more batch for them
                    index1=batches*batch # get the start index of the last batch
                    index2=batch-(shape0-batches*batch) # get the end index of the last batch
                    data_batch=tf.concat([test_data[index1:],test_data[:index2]],0) # concatenate the remaining samples and some samples from the beginning of testing data to form the last batch
                    labels_batch=tf.concat([test_labels[index1:],test_labels[:index2]],0) # concatenate the corresponding labels to form the last batch
                    batch_loss,batch_acc=self.test_(data_batch,labels_batch) # perform one testing step with data and labels batch as inputs
                    total_loss+=batch_loss # accumulate the total loss value
                    if hasattr(self.nn,'accuracy'): 
                        total_acc+=batch_acc # accumulate the total accuracy value if using accuracy metric
                test_loss=total_loss.numpy()/batches # calculate the average testing loss value
                if hasattr(self.nn,'accuracy'): 
                    test_acc=total_acc.numpy()/batches # calculate the average testing accuracy value if using accuracy metric
        else: 
            batch_loss,batch_acc=self.test_(test_data,test_labels) # perform one testing step with testing data and labels as inputs
            test_loss=test_loss.numpy() # convert the testing loss value to a numpy array
            if hasattr(self.nn,'accuracy'): 
                test_acc=test_acc.numpy() # convert the testing accuracy value to a numpy array if using accuracy metric
        if hasattr(self.nn,'accuracy'): 
            return test_loss,test_acc # return the testing loss and accuracy values if using accuracy metric
        else: 
            return test_loss # return the testing loss value if not using accuracy metric
