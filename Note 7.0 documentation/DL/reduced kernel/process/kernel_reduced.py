import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Value,Array
import numpy as np
import matplotlib.pyplot as plt


class kernel:
    def __init__(self,nn=None):
        self.nn=nn # the neural network object
        try:
            self.nn.km=1 # a flag to indicate the kernel mode
        except Exception:
            pass
        self.PO=None # the order of the optimizer
        self.process=None # the number of processes
        self.train_ds=None # the training dataset
        self.data_segment_flag=False # a flag to indicate whether to segment the data
        self.batches=None # the number of batches
        self.buffer_size=None # the buffer size for shuffling the data
        self.priority_flag=False # a flag to indicate whether to use priority for optimization
        self.priority_p=0 # the priority parameter
        self.max_opt=None # the maximum number of optimization steps
        self.epoch=None # the number of epochs
        self.epoch_counter=0 # the epoch counter
        self.stop=False # a flag to indicate whether to stop the training
        self.stop_flag=False # a flag to indicate whether to stop the training by condition
        self.save_flag=False # a flag to indicate whether to save the model
        self.batch=None # the batch size
        self.end_loss=None # the end condition for training loss
        self.end_acc=None # the end condition for training accuracy
        self.end_test_loss=None # the end condition for test loss
        self.end_test_acc=None # the end condition for test accuracy
        self.acc_flag='%' # the format for displaying accuracy
        self.opt_counter=None # the counter for optimization steps
        self.train_loss=0 # the training loss
        self.train_acc=0 # the training accuracy
        self.train_loss_list=[] # the list of training loss values
        self.train_acc_list=[] # the list of training accuracy values
        self.test_loss=0 # the test loss
        self.test_acc=0 # the test accuracy
        self.test_loss_list=[] # the list of test loss values
        self.test_acc_list=[] # the list of test accuracy values
        self.test_flag=False # a flag to indicate whether to use test data
        self.total_epoch=0 # the total number of epochs
    
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        if type(self.nn.param[0])!=list:
            self.train_data=train_data.astype(self.nn.param[0].dtype.name) # convert the train data type to match the model parameter type
            self.train_labels=train_labels.astype(self.nn.param[0].dtype.name) # convert the train labels type to match the model parameter type
        else:
            self.train_data=train_data.astype(self.nn.param[0][0].dtype.name) # convert the train data type to match the model parameter type
            self.train_labels=train_labels.astype(self.nn.param[0][0].dtype.name) # convert the train labels type to match the model parameter type
        self.train_dataset=train_dataset # set the train dataset object
        if test_data is not None: 
            self.test_data=test_data  # set the test data array 
            self.test_labels=test_labels  # set the test labels array 
            self.test_flag=True  # set the test flag to True 
        self.test_dataset=test_dataset  # set the test dataset object 
        self.batch_counter=np.zeros(self.process,dtype=np.int32)  # initialize an array to count batches for each process 
        if type(self.nn.param[0])!=list:  # initialize an array to accumulate loss for each process 
            self.total_loss=np.zeros(self.process,dtype=self.nn.param[0].dtype.name)
        else:
            self.total_loss=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name)
        try:
            if self.nn.accuracy!=None:
                if type(self.nn.param[0])!=list:  # initialize an array to accumulate accuracy for each process 
                    self.total_acc=np.zeros(self.process,dtype=self.nn.param[0].dtype.name)
                else:
                    self.total_acc=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name)
        except Exception:
            pass
        if self.priority_flag==True:
            self.opt_counter=np.zeros(self.process,dtype=np.int32)  # initialize an array to count optimization steps for each process 
        if self.train_dataset==None:
            if type(self.train_data)==list:
                self.shape0=train_data[0].shape[0]  # get the number of samples in the first train data array 
                self.batches=int((self.shape0-self.shape0%self.batch)/self.batch)  # calculate the number of batches 
                if self.shape0%self.batch!=0:
                    self.batches+=1  # add one more batch if there are remaining samples 
            else:
                self.shape0=train_data.shape[0]  # get the number of samples in the train data array 
                self.batches=int((self.shape0-self.shape0%self.batch)/self.batch)  # calculate the number of batches 
                if self.shape0%self.batch!=0:
                    self.batches+=1  # add one more batch if there are remaining samples 
        if self.data_segment_flag==True:
            self.train_data,self.train_labels=self.segment_data()  # segment the train data and labels according to the number of processes 
        return
    
    
    def segment_data(self):
        if len(self.train_data)!=self.process:  # check if the train data is already segmented 
            segments=int((len(self.train_data)-len(self.train_data)%self.process)/self.process)  # calculate the number of samples for each segment 
            for i in range(self.process):  # loop over the processes 
                index1=i*segments  # get the start index of the segment 
                index2=(i+1)*segments  # get the end index of the segment 
                if i==0:  # for the first process 
                    data=np.expand_dims(self.train_data[index1:index2],axis=0)  # create a new dimension for the segment and assign it to data 
                    labels=np.expand_dims(self.train_labels[index1:index2],axis=0)  # create a new dimension for the segment and assign it to labels 
                else:  # for other processes 
                    data=np.concatenate((data,np.expand_dims(self.train_data[index1:index2],axis=0)))  # concatenate the segment to data along the new dimension 
                    labels=np.concatenate((labels,np.expand_dims(self.train_labels[index1:index2],axis=0)))  # concatenate the segment to labels along the new dimension 
            if len(data)%self.process!=0:  # check if there are remaining samples that are not segmented 
                segments+=1  # increase the number of samples for each segment by one 
                index1=segments*self.process  # get the start index of the remaining samples 
                index2=self.process-(len(self.train_data)-segments*self.process)  # get the number of processes that need to be filled with extra samples 
                data=np.concatenate((data,np.expand_dims(self.train_data[index1:index2],axis=0)))  # concatenate the remaining samples to data along the new dimension 
                labels=np.concatenate((labels,np.expand_dims(self.train_labels[index1:index2],axis=0)))  # concatenate the remaining samples to labels along the new dimension 
            return data,labels  # return the segmented data and labels
                
    
    def init(self,manager):
        self.epoch_counter=Value('i',self.epoch_counter)  # create a shared value for epoch counter
        self.batch_counter=Array('i',self.batch_counter)  # create a shared array for batch counter
        self.total_loss=Array('f',self.total_loss)  # create a shared array for total loss
        self.total_epoch=Value('i',self.total_epoch)  # create a shared value for total epoch
        self.train_loss=Value('f',self.train_loss)  # create a shared value for train loss
        self.train_loss_list=manager.list(self.train_loss_list)  # create a shared list for train loss values
        self.priority_p=Value('i',self.priority_p)  # create a shared value for priority parameter
        if self.test_flag==True:
            self.test_loss=Value('f',self.test_loss)  # create a shared value for test loss
            self.test_loss_list=manager.list(self.test_loss_list)  # create a shared list for test loss values
        try:
            if self.nn.accuracy!=None:
                self.total_acc=Array('f',self.total_acc)  # create a shared array for total accuracy
                self.train_acc=Value('f',self.train_acc)  # create a shared value for train accuracy
                self.train_acc_list=manager.list(self.train_acc_list)  # create a shared list for train accuracy values
                if self.test_flag==True:
                    self.test_acc=Value('f',self.test_acc)  # create a shared value for test accuracy
                    self.test_acc_list=manager.list(self.test_acc_list)  # create a shared list for test accuracy values
        except Exception:
            pass
        if self.priority_flag==True:
            self.opt_counter=Array('i',self.opt_counter)  # create a shared array for optimization counter 
        try:
            if self.nn.attenuate!=None:
              self.nn.opt_counter=manager.list([self.nn.opt_counter])  # create a shared list for the neural network's optimization counter 
        except Exception:
            pass
        try:
            self.nn.ec=manager.list([self.nn.ec])  # create a shared list for the neural network's epoch counter 
        except Exception:
            pass
        try:
            self.nn.bc=manager.list([self.nn.bc])  # create a shared list for the neural network's batch counter 
        except Exception:
            pass
        self.stop_flag=Value('b',self.stop_flag)  # create a shared value for stop flag 
        self.save_flag=Value('b',self.save_flag)  # create a shared value for save flag 
        self.param=manager.dict()  # create a shared dictionary for parameters 
        self.param[7]=self.nn.param  # assign the neural network's parameters to the dictionary 
        return
    
    
    def end(self):
        if self.end_loss!=None and len(self.train_loss_list)!=0 and self.train_loss_list[-1]<self.end_loss:  # check if the train loss is lower than the end condition 
            return True
        elif self.end_acc!=None and len(self.train_acc_list)!=0 and self.train_acc_list[-1]>self.end_acc:  # check if the train accuracy is higher than the end condition 
            return True
        elif self.end_loss!=None and len(self.train_loss_list)!=0 and self.end_acc!=None and self.train_loss_list[-1]<self.end_loss and self.train_acc_list[-1]>self.end_acc:  # check if both the train loss and accuracy meet the end condition 
            return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.test_loss_list[-1]<self.end_test_loss:  # check if the test loss is lower than the end condition 
            return True
        elif self.end_test_acc!=None and len(self.test_acc_list)!=0 and self.test_acc_list[-1]>self.end_test_acc:  # check if the test accuracy is higher than the end condition 
            return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.end_test_acc!=None and self.test_loss_list[-1]<self.end_test_loss and self.test_acc_list[-1]>self.end_test_acc:  # check if both the test loss and accuracy meet the end condition 
            return True
    
    
    @tf.function
    def opt_p(self,data,labels,p,lock,g_lock=None):
        try:
            try:
                if self.nn.GradientTape!=None:  # check if the neural network has its own gradient tape function 
                    tape,output,loss=self.nn.GradientTape(data,labels,p)  # use the neural network's gradient tape function to get the tape, output and loss 
            except Exception:  
                with tf.GradientTape(persistent=True) as tape:  # use the default gradient tape function 
                    try:
                        try:
                            output=self.nn.fp(data)  # use the neural network's forward propagation function to get the output 
                            loss=self.nn.loss(output,labels)  # use the neural network's loss function to get the loss 
                        except Exception:
                            output,loss=self.nn.fp(data,labels)  # use the neural network's forward propagation function to get the output and loss 
                    except Exception:
                        try:
                            output=self.nn.fp(data,p)  # use the neural network's forward propagation function to get the output with the process number 
                            loss=self.nn.loss(output,labels)  # use the neural network's loss function to get the loss 
                        except Exception:
                            output,loss=self.nn.fp(data,labels,p)  # use the neural network's forward propagation function to get the output and loss with the process number 
        except Exception as e:
            raise e
        if self.PO==1:  # check if the optimizer order is 1 (lock before gradient calculation)
            if self.priority_flag==True and self.priority_p.value!=-1:  # check if the priority flag is True and the priority parameter is not -1
                while True:
                    if p==self.priority_p.value:  # check if the process number matches the priority parameter
                        break
                    else:
                        continue
            lock[0].acquire()  # acquire the first lock
            if self.stop_func_(lock[0]):  # check if the stop condition is met
                return None,0
            try:
                try:
                    try:
                        gradient=self.nn.gradient(tape,loss)  # use the neural network's gradient function to get the gradient 
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])  # use the neural network's gradient function to get the gradient with the parameters 
                except Exception:
                    gradient=tape.gradient(loss,self.nn.param)  # use the default gradient function to get the gradient 
            except Exception as e:
                raise e
            try:
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)  # use the neural network's attenuate function to modify the gradient 
            except Exception as e:
                try:
                    if self.nn.attenuate!=None:  # check if the neural network has an attenuate function 
                        raise e
                except Exception:
                    pass
            try:
                try:
                    param=self.nn.opt(gradient)  # use the neural network's optimizer function to update the parameters 
                except Exception:
                    param=self.nn.opt(gradient,p)  # use the neural network's optimizer function to update the parameters with the process number 
            except Exception as e:
                raise e
            lock[0].release()  # release the first lock
        elif self.PO==2:  # check if the optimizer order is 2 (lock after gradient calculation)
            g_lock.acquire()  # acquire the global lock
            if self.stop_func_(g_lock):  # check if the stop condition is met
                return None,0
            try:
                try:
                    try:
                        gradient=self.nn.gradient(tape,loss)  # use the neural network's gradient function to get the gradient 
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])  # use the neural network's gradient function to get the gradient with the parameters 
                except Exception:
                    gradient=tape.gradient(loss,self.nn.param)  # use the default gradient function to get the gradient 
            except Exception as e:
                raise e
            g_lock.release()  # release the global lock
            if self.priority_flag==True and self.priority_p.value!=-1:  # check if the priority flag is True and the priority parameter is not -1
                while True:
                    if p==self.priority_p.value:  # check if the process number matches the priority parameter
                        break
                    else:
                        continue
            lock[0].acquire()  # acquire the first lock
            if self.stop_func_(lock[0]):  # check if the stop condition is met
                return None,0
            try:
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)  # use the neural network's attenuate function to modify the gradient 
            except Exception as e:
                try:
                    if self.nn.attenuate!=None:  # check if the neural network has an attenuate function 
                        raise e
                except Exception:
                    pass
            try:
                try:
                    param=self.nn.opt(gradient)  # use the neural network's optimizer function to update the parameters 
                except Exception:
                    param=self.nn.opt(gradient,p)  # use the neural network's optimizer function to update the parameters with the process number 
            except Exception as e:
                raise e
            lock[0].release()  # release the first lock
        elif self.PO==3:  # check if the optimizer order is 3 (no lock)
            if self.priority_flag==True and self.priority_p.value!=-1:  # check if the priority flag is True and the priority parameter is not -1
                while True:
                    if p==self.priority_p.value:  # check if the process number matches the priority parameter
                        break
                    else:
                        continue
            if self.stop_func_():  # check if the stop condition is met
                return None,0
            try:
                try:
                    try:
                        gradient=self.nn.gradient(tape,loss)  # use the neural network's gradient function to get the gradient 
                    except Exception:
                        gradient=self.nn.gradient(tape,loss,self.param[7])  # use the neural network's gradient function to get the gradient with the parameters 
                except Exception:
                    gradient=tape.gradient(loss,self.nn.param)  # use the default gradient function to get the gradient 
            except Exception as e:
                raise e
            try:
                gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)  # use the neural network's attenuate function to modify the gradient 
            except Exception as e:
                try:
                    if self.nn.attenuate!=None:  # check if the neural network has an attenuate function 
                        raise e
                except Exception:
                    pass
            try:
                try:
                    param=self.nn.opt(gradient)  # use the neural network's optimizer function to update the parameters 
                except Exception:
                    param=self.nn.opt(gradient,p)  # use the neural network's optimizer function to update the parameters with the process number 
            except Exception as e:
                raise e
        return output,loss,param  # return output, loss and parameters
    
    
    def opt(self,data,labels,p,lock,g_lock):
        if self.PO==2:  # check if the optimizer order is 2 (lock after gradient calculation)
            if type(g_lock)!=list:  # check if g_lock is not a list 
                pass
            elif len(g_lock)==self.process:  # check if g_lock has same length as process number 
                ln=p  # assign process number to ln (local number)
                g_lock=g_lock[ln]  # assign g_lock element at ln index to g_lock 
            else:  
                ln=int(np.random.choice(len(g_lock)))  # assign a random integer from g_lock length to ln (local number)
                g_lock=g_lock[ln]  # assign g_lock element at ln index to g_lock 
            output,loss,param=self.opt_p(data,labels,p,lock,g_lock)  # call the opt_p function to get output, loss and parameters 
        else:
            output,loss,param=self.opt_p(data,labels,p,lock)  # call the opt_p function to get output, loss and parameters without g_lock 
        return output,loss,param  # return output, loss and parameters
    
    
    def update_nn_param(self,param=None):
        if param==None:  # check if param is None 
            parameter_flat=nest.flatten(self.nn.param)  # flatten the neural network's parameters 
            parameter7_flat=nest.flatten(self.param[7])  # flatten the kernel's parameters 
        else:
            parameter_flat=nest.flatten(self.nn.param)  # flatten the neural network's parameters 
            parameter7_flat=nest.flatten(param)  # flatten the given param 
        for i in range(len(parameter_flat)):  # loop over the flattened parameters 
            if param==None:  # check if param is None 
                state_ops.assign(parameter_flat[i],parameter7_flat[i])  # assign the kernel's parameters to the neural network's parameters 
            else:
                state_ops.assign(parameter_flat[i],parameter7_flat[i])  # assign the given param to the neural network's parameters 
        self.nn.param=nest.pack_sequence_as(self.nn.param,parameter_flat)  # pack the flattened parameters back to the neural network's parameters 
        self.param[7]=nest.pack_sequence_as(self.param[7],parameter7_flat)  # pack the flattened parameters back to the kernel's parameters 
        return
    
    
    def train7(self,train_ds,p,test_batch,lock,g_lock):
        while True:  # loop until break
            for data_batch,labels_batch in train_ds:  # loop over the train dataset batches
                try:
                    data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)  # use the neural network's data function to process data and labels
                except Exception as e:
                    try:
                        if self.nn.data_func!=None:  # check if the neural network has a data function
                            raise e
                    except Exception:
                        pass
                if self.priority_flag==True:  # check if the priority flag is True
                    self.priority_p.value=np.argmax(self.opt_counter)  # assign the index of the maximum value in opt_counter to priority parameter
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt:  # check if max_opt is not None and opt_counter at priority parameter index is greater than or equal to max_opt
                        self.priority_p.value=int(self.priority_p.value)  # convert priority parameter to integer type
                    elif self.max_opt==None:  # check if max_opt is None
                        self.priority_p.value=int(self.priority_p.value)  # convert priority parameter to integer type
                    else:
                        self.priority_p.value=-1  # assign -1 to priority parameter
                if self.priority_flag==True:  # check if the priority flag is True
                    self.opt_counter[p]=0  # assign zero to opt_counter at process number
                try:
                    opt_counter=self.nn.opt_counter[0]  # get the neural network's optimization counter
                    opt_counter.scatter_update(tf.IndexedSlices(0,p))  # update opt_counter with zero at process number
                    self.nn.opt_counter[0]=opt_counter  # assign opt_counter back to neural network's optimization counter
                except Exception as e:
                    try:
                       if self.nn.attenuate!=None:  # check if the neural network has an attenuate function
                           raise e
                    except Exception:
                        pass
                output,batch_loss,param=self.opt(data_batch,labels_batch,p,lock,g_lock)  # call the opt function to get output, batch loss and parameters
                self.param[7]=param  # assign param to kernel's parameters
                if self.priority_flag==True:  # check if the priority flag is True
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')  # get a numpy array from opt_counter shared array object
                    opt_counter+=1  # increment opt_counter by one for each element
                try:
                    opt_counter=self.nn.opt_counter[0]  # get the neural network's optimization counter
                    opt_counter.assign(opt_counter+1)  # increment opt_counter by one
                    self.nn.opt_counter[0]=opt_counter  # assign opt_counter back to neural network's optimization counter
                except Exception as e:
                    try:
                       if self.nn.attenuate!=None:  # check if the neural network has an attenuate function
                           raise e
                    except Exception:
                        pass
                try:
                    bc=self.nn.bc[0]  # get the neural network's batch counter
                    bc.assign_add(1)  # increment bc by one
                    self.nn.bc[0]=bc  # assign bc back to neural network's batch counter
                except Exception:
                    pass
                try:
                    batch_acc=self.nn.accuracy(output,labels_batch)  # use the neural network's accuracy function to get the batch accuracy
                except Exception as e:
                    try:
                        if self.nn.accuracy!=None:  # check if the neural network has an accuracy function
                           raise e
                    except Exception:
                       pass
                try:
                    if self.nn.accuracy!=None:  # check if the neural network has an accuracy function
                        self.total_loss[p]+=batch_loss  # accumulate batch loss to total loss at process number
                        self.total_acc[p]+=batch_acc  # accumulate batch accuracy to total accuracy at process number
                except Exception:
                    self.total_loss[p]+=batch_loss  # accumulate batch loss to total loss at process number
                self.batch_counter[p]+=1  # increment batch counter at process number
                if self.PO==1 or self.PO==2:  # check if the optimizer order is 1 or 2 (lock before or after gradient calculation)
                    lock[1].acquire()  # acquire the second lock
                elif lock!=None:  # check if lock is not None
                    lock.acquire()  # acquire the lock 
                batches=np.sum(self.batch_counter)  # sum up the batch counter for all processes 
                if batches>=self.batches:  # check if the number of batches reaches the total number of batches 
                    batch_counter=np.frombuffer(self.batch_counter.get_obj(),dtype='i')  # get a numpy array from batch counter shared array object 
                    batch_counter*=0  # reset batch counter to zero for each element 
                    loss=np.sum(self.total_loss)/batches  # calculate the average loss for all batches 
                    try:
                        if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                            train_acc=np.sum(self.total_acc)/batches  # calculate the average accuracy for all batches 
                    except Exception:
                        pass
                    self.total_epoch.value+=1  # increment total epoch by one 
                    self.train_loss.value=loss  # assign loss to train loss 
                    self.train_loss_list.append(loss)  # append loss to train loss list 
                    try:
                        if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                            self.train_acc.value=train_acc  # assign train_acc to train accuracy 
                            self.train_acc_list.append(train_acc)  # append train_acc to train accuracy list 
                    except Exception:
                        pass
                    if self.test_flag==True:  # check if the test flag is True 
                        self.test_loss.value,self.test_acc.value=self.test(self.test_data,self.test_labels,test_batch,p)  # call the test function to get test loss and accuracy 
                        self.test_loss_list.append(self.test_loss.value)  # append test loss to test loss list 
                    try:
                        if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                            self.test_acc_list.append(self.test_acc.value)  # append test accuracy to test accuracy list 
                    except Exception:
                        pass
                    self.print_save()  # call the print_save function to print and save the results 
                    self.epoch_counter.value+=1  # increment epoch counter by one 
                    try:
                        ec=self.nn.ec[0]  # get the neural network's epoch counter
                        ec.assign_add(1)  # increment ec by one
                        self.nn.ec[0]=ec  # assign ec back to neural network's epoch counter
                    except Exception:
                        pass
                    total_loss=np.frombuffer(self.total_loss.get_obj(),dtype='f')  # get a numpy array from total loss shared array object
                    total_loss*=0  # reset total loss to zero for each element
                    try:
                        total_acc=np.frombuffer(self.total_acc.get_obj(),dtype='f')  # get a numpy array from total accuracy shared array object
                        total_acc*=0  # reset total accuracy to zero for each element
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:  # check if the neural network has an accuracy function
                                raise e
                        except Exception:
                            pass
                if self.PO==1 or self.PO==2:  # check if the optimizer order is 1 or 2 (lock before or after gradient calculation)
                    lock[1].release()  # release the second lock
                elif lock!=None:  # check if lock is not None
                    lock.release()  # release the lock 
                if self.epoch_counter.value>=self.epoch:  # check if the epoch counter reaches the epoch number 
                    self.param[7]=param  # assign param to kernel's parameters 
                    return
    
    
    def train(self,p,lock=None,g_lock=None,test_batch=None):
        if self.epoch!=None:  # check if epoch is not None 
            if self.train_dataset!=None:  # check if train dataset is not None 
                train_ds=self.train_dataset  # assign train dataset to train_ds 
            else:
                if self.data_segment_flag==True:  # check if data segment flag is True 
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data[p],self.train_labels[p])).batch(self.batch)  # create a dataset from tensor slices of segmented data and labels and batch them 
                elif self.buffer_size!=None:  # check if buffer size is not None 
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).shuffle(self.buffer_size).batch(self.batch)  # create a dataset from tensor slices of data and labels and shuffle and batch them with buffer size 
                else:
                    train_ds=tf.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).batch(self.batch)  # create a dataset from tensor slices of data and labels and batch them 
        self.train7(train_ds,p,test_batch,lock,g_lock)  # call the train7 function with train_ds, process number, test batch, lock and g_lock 
        return
    
    
    def test(self,test_data=None,test_labels=None,batch=None,p=None):
        if type(self.nn.param[0])!=list:  
            test_data=test_data.astype(self.nn.param[0].dtype.name)  # convert the test data type to match the model parameter type
            test_labels=test_labels.astype(self.nn.param[0].dtype.name)  # convert the test labels type to match the model parameter type
        else:
            test_data=test_data.astype(self.nn.param[0][0].dtype.name)  # convert the test data type to match the model parameter type
            test_labels=test_labels.astype(self.nn.param[0][0].dtype.name)  # convert the test labels type to match the model parameter type
        if batch!=None:  # check if batch is not None 
            total_loss=0  # initialize total loss to zero 
            total_acc=0  # initialize total accuracy to zero 
            if self.test_dataset!=None:  # check if test dataset is not None 
                for data_batch,labels_batch in self.test_dataset:  # loop over the test dataset batches 
                    try:
                        try:
                            output=self.nn.fp(data_batch)  # use the neural network's forward propagation function to get the output 
                        except Exception:
                            output=self.nn.fp(data_batch,p)  # use the neural network's forward propagation function to get the output with the process number 
                    except Exception as e:
                        raise e
                    batch_loss=self.nn.loss(output,labels_batch)  # use the neural network's loss function to get the batch loss 
                    total_loss+=batch_loss  # accumulate batch loss to total loss 
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch)  # use the neural network's accuracy function to get the batch accuracy 
                        total_acc+=batch_acc  # accumulate batch accuracy to total accuracy 
                    except Exception as e:
                        try:
                           if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                               raise e
                        except Exception:
                            pass
            else:  # if test dataset is None 
                total_loss=0  # initialize total loss to zero 
                total_acc=0  # initialize total accuracy to zero 
                batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)  # calculate the number of batches 
                shape0=test_data.shape[0]  # get the number of samples in the test data array 
                for j in range(batches):  # loop over the batches 
                    index1=j*batch  # get the start index of the batch 
                    index2=(j+1)*batch  # get the end index of the batch 
                    data_batch=test_data[index1:index2]  # get the data batch from test data array 
                    if type(test_labels)==list:  # check if test labels is a list 
                        for i in range(len(test_labels)):  
                            labels_batch[i]=test_labels[i][index1:index2]  # get the labels batch from test labels list 
                    else:
                        labels_batch=test_labels[index1:index2]  # get the labels batch from test labels array 
                    try:
                        try:
                            output=self.nn.fp(data_batch)  # use the neural network's forward propagation function to get the output 
                        except Exception:
                            output=self.nn.fp(data_batch,p)  # use the neural network's forward propagation function to get the output with the process number
                    except Exception as e:
                        raise e
                    batch_loss=self.nn.loss(output,labels_batch)  # use the neural network's loss function to get the batch loss 
                    total_loss+=batch_loss  # accumulate batch loss to total loss 
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch)  # use the neural network's accuracy function to get the batch accuracy 
                        total_acc+=batch_acc  # accumulate batch accuracy to total accuracy 
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                                raise e
                        except Exception:
                            pass
                if shape0%batch!=0:  # check if there are remaining samples that are not in a batch 
                    batches+=1  # increment batches by one 
                    index1=batches*batch  # get the start index of the remaining samples 
                    index2=batch-(shape0-batches*batch)  # get the number of samples that need to be filled with extra samples 
                    data_batch=tf.concat([test_data[index1:],test_data[:index2]],0)  # concatenate the remaining samples and extra samples to form a data batch 
                    labels_batch=tf.concat([test_labels[index1:],test_labels[:index2]],0)  # concatenate the remaining labels and extra labels to form a labels batch 
                    try:
                        try:
                            output=self.nn.fp(data_batch)  # use the neural network's forward propagation function to get the output 
                        except Exception:
                            output=self.nn.fp(data_batch,p)  # use the neural network's forward propagation function to get the output with the process number
                    except Exception as e:
                        raise e
                    batch_loss=self.nn.loss(output,labels_batch)  # use the neural network's loss function to get the batch loss 
                    total_loss+=batch_loss  # accumulate batch loss to total loss 
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch)  # use the neural network's accuracy function to get the batch accuracy 
                        total_acc+=batch_acc  # accumulate batch accuracy to total accuracy 
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                                raise e
                        except Exception:
                            pass
            test_loss=total_loss.numpy()/batches  # calculate the average test loss for all batches 
            try:
                if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                    test_acc=total_acc.numpy()/batches  # calculate the average test accuracy for all batches 
            except Exception:
                pass
        else:  # if batch is None 
            try:
                try:
                    output=self.nn.fp(test_data)  # use the neural network's forward propagation function to get the output 
                except Exception:
                    output=self.nn.fp(test_data,p)  # use the neural network's forward propagation function to get the output with the process number
            except Exception as e:
                raise e
            test_loss=self.nn.loss(output,test_labels)  # use the neural network's loss function to get the test loss 
            test_loss=test_loss.numpy()  # convert test loss to numpy array 
            try:
                test_acc=self.nn.accuracy(output,test_labels)  # use the neural network's accuracy function to get the test accuracy 
                test_acc=test_acc.numpy()  # convert test accuracy to numpy array 
            except Exception as e:
                try:
                    if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                        raise e
                except Exception:
                    pass
        try:
            if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                return test_loss,test_acc  # return test loss and accuracy 
        except Exception:
            return test_loss,None  # return test loss and None
    
    
    def stop_func(self):
        if self.end():  # check if any of the end conditions is met
            self.save(self.total_epoch.value,True)  # save the model with total epoch and True flag
            self.save_flag.value=True  # set save flag to True
            self.stop_flag.value=True  # set stop flag to True
            return True
        return False
    
    
    def stop_func_(self,lock=None):
        if self.stop==True:  # check if stop is True
            if self.stop_flag.value==True or self.stop_func():  # check if the stop flag is True or the stop function returns True
                if self.PO!=3:  # check if the optimizer order is not 3 (no lock)
                    lock.release()  # release the lock
                return True
        return False
    
    
    def visualize_train(self):
        print()  # print a blank line
        plt.figure(1)  # create a new figure
        plt.plot(np.arange(self.total_epoch.value),self.train_loss_list)  # plot the train loss list against the total epoch
        plt.title('train loss')  # set the title of the figure
        plt.xlabel('epoch')  # set the x-axis label of the figure
        plt.ylabel('loss')  # set the y-axis label of the figure
        print('train loss:{0:.6f}'.format(self.train_loss.value))  # print the train loss value with six decimal places
        try:
            if self.nn.accuracy!=None:  # check if the neural network has an accuracy function 
                plt.figure(2)  # create a new figure
                plt.plot(np.arange(self.total_epoch.value),self.train_acc_list)  # plot the train accuracy list against the total epoch
                plt.title('train acc')  # set the title of the figure
                plt.xlabel('epoch')  # set the x-axis label of the figure
                plt.ylabel('acc')  # set the y-axis label of the figure
                if self.acc_flag=='%':  # check if the accuracy format is percentage 
                    print('train acc:{0:.1f}'.format(self.train_acc.value*100))  # print the train accuracy value with one decimal place and percentage sign 
                else:
                    print('train acc:{0:.6f}'.format(self.train_acc.value))  # print the train accuracy value with six decimal places 
        except Exception:
            pass
        return