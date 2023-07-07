from tensorflow import function
import numpy as np
import matplotlib.pyplot as plt
import time


class kernel:
    def __init__(self,nn=None):
        self.nn=nn # the neural network object
        try:
            self.nn.km=1 # a flag to indicate the kernel mode
        except Exception:
            pass
        self.platform=None # the platform to use, either tensorflow or pytorch
        self.batches=None # the number of batches for training data
        self.suspend=False # a flag to indicate whether to suspend the training
        self.stop=False # a flag to indicate whether to stop the training
        self.stop_flag=False # a flag to indicate whether the training has been stopped
        self.save_epoch=None # the epoch number to save the model
        self.batch=None # the batch size for training data
        self.epoch=0 # the current epoch number
        self.end_loss=None # the target loss value to end the training
        self.end_acc=None # the target accuracy value to end the training
        self.end_test_loss=None # the target test loss value to end the training
        self.end_test_acc=None # the target test accuracy value to end the training
        self.acc_flag='%' # a flag to indicate whether to use percentage or decimal for accuracy display
        self.train_counter=0 # a counter for how many times the train method has been called
        self.filename='save.dat' # the file name to save the model
        self.train_loss=None # the current train loss value
        self.train_acc=None # the current train accuracy value
        self.train_loss_list=[] # a list of train loss values for each epoch
        self.train_acc_list=[] # a list of train accuracy values for each epoch
        self.test_loss=None # the current test loss value
        self.test_acc=None # the current test accuracy value
        self.test_loss_list=[] # a list of test loss values for each epoch
        self.test_acc_list=[] # a list of test accuracy values for each epoch
        self.test_flag=False # a flag to indicate whether to use test data or not
        self.total_epoch=0 # the total number of epochs for all trainings
        self.time=0 # the time elapsed for one training session
        self.total_time=0 # the total time elapsed for all trainings
    
    
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        if type(self.nn.param[0])!=list:
            self.train_data=train_data.astype(self.nn.param[0].dtype.name) # convert the train data type to match the neural network parameters type
            self.train_labels=train_labels.astype(self.nn.param[0].dtype.name) # convert the train labels type to match the neural network parameters type
        else:
            self.train_data=train_data.astype(self.nn.param[0][0].dtype.name) # convert the train data type to match the neural network parameters type (for multiple inputs)
            self.train_labels=train_labels.astype(self.nn.param[0][0].dtype.name) # convert the train labels type to match the neural network parameters type (for multiple inputs)
        self.train_dataset=train_dataset # a tensorflow or pytorch dataset object for train data (optional)
        if test_data is not None: 
            if type(self.nn.param[0])!=list:
                self.test_data=test_data.astype(self.nn.param[0].dtype.name) # convert the test data type to match the neural network parameters type
                self.test_labels=test_labels.astype(self.nn.param[0].dtype.name) # convert the test labels type to match the neural network parameters type
            else:
                self.test_data=test_data.astype(self.nn.param[0][0].dtype.name)  # convert the test data type to match the neural network parameters type (for multiple inputs)
                self.test_labels=test_labels.astype(self.nn.param[0][0].dtype.name)  # convert the test labels type to match the neural network parameters type (for multiple inputs)
            self.test_flag=True  # set the test flag to True if test data is provided 
        self.test_dataset=test_dataset  # a tensorflow or pytorch dataset object for test data (optional)
        if self.train_dataset==None: 
            self.shape0=train_data.shape[0]  # get the number of samples in train data
        return
    
    
    def init(self): # a method to initialize the attributes for a new training session
        self.suspend=False 
        self.stop=False 
        self.stop_flag=False 
        self.save_epoch=None 
        self.end_loss=None 
        self.end_acc=None 
        self.end_test_loss=None 
        self.end_test_acc=None 
        self.train_loss=None 
        self.train_acc=None 
        self.test_loss=None 
        self.test_acc=None 
        self.train_loss_list.clear() 
        self.train_acc_list.clear() 
        self.test_loss_list.clear() 
        self.test_acc_list.clear() 
        self.test_flag=False 
        self.train_counter=0 
        self.epoch=0 
        self.total_epoch=0 
        self.time=0 
        self.total_time=0
        return
    
    
    def end(self): # a method to check whether the training has reached the target loss or accuracy values
        if self.end_loss!=None and len(self.train_loss_list)!=0 and self.train_loss_list[-1]<self.end_loss: # if the target train loss is given and the current train loss is lower than it
            return True
        elif self.end_acc!=None and len(self.train_acc_list)!=0 and self.train_acc_list[-1]>self.end_acc: # if the target train accuracy is given and the current train accuracy is higher than it
            return True
        elif self.end_loss!=None and len(self.train_loss_list)!=0 and self.end_acc!=None and self.train_loss_list[-1]<self.end_loss and self.train_acc_list[-1]>self.end_acc: # if both the target train loss and accuracy are given and the current train loss is lower than it and the current train accuracy is higher than it
            return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.test_loss_list[-1]<self.end_test_loss: # if the target test loss is given and the current test loss is lower than it
            return True
        elif self.end_test_acc!=None and len(self.test_acc_list)!=0 and self.test_acc_list[-1]>self.end_test_acc: # if the target test accuracy is given and the current test accuracy is higher than it
            return True
        elif self.end_test_loss!=None and len(self.test_loss_list)!=0 and self.end_test_acc!=None and self.test_loss_list[-1]<self.end_test_loss and self.test_acc_list[-1]>self.end_test_acc: # if both the target test loss and accuracy are given and the current test loss is lower than it and the current test accuracy is higher than it
            return True
    
    
    def loss_acc(self,output=None,labels_batch=None,loss=None,test_batch=None,total_loss=None,total_acc=None): # a method to calculate the loss and accuracy values for each batch or epoch
        if self.batch!=None: # if batch mode is used
            total_loss+=loss # accumulate the batch loss to total loss
            try:
                batch_acc=self.nn.accuracy(output,labels_batch) # calculate the batch accuracy using the neural network's accuracy method
                total_acc+=batch_acc # accumulate the batch accuracy to total accuracy
            except Exception as e:
                try:
                    if self.nn.accuracy!=None: # if the neural network has an accuracy method defined
                        raise e # raise the exception
                except Exception:
                    pass
            return total_loss,total_acc # return the total loss and accuracy values for all batches so far
        else: # if batch mode is not used (use all data at once)
            loss=loss.numpy() # convert the loss value to numpy array
            self.train_loss=loss # assign the loss value to train loss attribute
            self.train_loss_list.append(loss) # append the loss value to train loss list
            try:
                acc=self.nn.accuracy(output,self.train_labels) # calculate the accuracy value using the neural network's accuracy method
                acc=acc.numpy() # convert the accuracy value to numpy array
                self.train_acc=acc # assign the accuracy value to train accuracy attribute
                self.train_acc_list.append(acc) # append the accuracy value to train accuracy list
            except Exception as e:
                try:
                    if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined
                        raise e  # raise the exception
                except Exception:
                    pass
            if self.test_flag==True:  # if test data is used
                self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)  # calculate the test loss and accuracy values using the test method
                self.test_loss_list.append(self.test_loss)  # append the test loss value to test loss list
                try:
                    self.test_acc_list.append(self.test_acc)  # append the test accuracy value to test accuracy list
                except Exception as e:
                    try:
                        if self.nn.accuracy!=None: # if the neural network has an accuracy method defined
                            raise e # raise the exception
                    except Exception:
                        pass
            return # return nothing for this case
    
    
    def data_func(self,data_batch=None,labels_batch=None,batch=None,index1=None,index2=None,j=None,flag=None): # a method to get a batch of data and labels from the train data and labels
        if flag==None: # if flag is None, it means the batch size is smaller than the number of samples
            if batch!=1: # if batch size is not 1
                data_batch=self.train_data[index1:index2] # get a slice of train data according to the index range
            else: # if batch size is 1
                data_batch=self.train_data[j] # get one sample of train data according to the index
            if batch!=1: # if batch size is not 1
                labels_batch=self.train_labels[index1:index2] # get a slice of train labels according to the index range
            else: # if batch size is 1
                labels_batch=self.train_labels[j] # get one sample of train labels according to the index
        else: # if flag is not None, it means the batch size is larger than the number of samples
            try:
                try:
                    data_batch=self.platform.concat([self.train_data[index1:],self.train_data[:index2]],0) # concatenate two slices of train data from the end and the beginning to form a batch
                    labels_batch=self.platform.concat([self.train_labels[index1:],self.train_labels[:index2]],0) # concatenate two slices of train labels from the end and the beginning to form a batch
                except Exception: # if the platform's concat method fails
                    data_batch=np.concatenate([self.train_data[index1:],self.train_data[:index2]],0) # use numpy's concatenate method instead
                    labels_batch=np.concatenate([self.train_labels[index1:],self.train_labels[:index2]],0) # use numpy's concatenate method instead
            except Exception as e:
                raise e # raise any other exception
        return data_batch,labels_batch # return the batch of data and labels
    
    
    @function(jit_compile=True) # use tensorflow's function decorator to speed up the execution
    def tf_opt(self,data,labels): # a method to perform one optimization step using tensorflow platform
        try:
            try:
                if self.nn.GradientTape!=None: # if the neural network has a GradientTape method defined
                    tape,output,loss=self.nn.GradientTape(data,labels) # use the neural network's GradientTape method to get the tape, output and loss values
            except Exception: # if the neural network does not have a GradientTape method defined or it fails
                with self.platform.GradientTape(persistent=True) as tape: # use tensorflow's GradientTape context manager instead
                    try:
                        output=self.nn.fp(data) # get the output value using the neural network's forward propagation method
                        loss=self.nn.loss(output,labels) # get the loss value using the neural network's loss function
                    except Exception: # if the neural network's forward propagation method or loss function fails or they are combined in one method 
                        output,loss=self.nn.fp(data,labels) # use the neural network's forward propagation method with both data and labels as inputs to get the output and loss values 
        except Exception as e:
            raise e # raise any other exception 
        try:
            try:
                gradient=self.nn.gradient(tape,loss) # use the neural network's gradient method to get the gradient value from the tape and loss values 
            except Exception: # if the neural network does not have a gradient method defined or it fails 
                gradient=tape.gradient(loss,self.nn.param)  # use tensorflow's tape.gradient method instead with loss value and neural network parameters as inputs 
        except Exception as e:
            raise e  # raise any other exception 
        try:
            try:
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))  # use the neural network's optimizer's apply_gradients method to update the neural network parameters with gradient value 
            except Exception:  # if the neural network does not have an optimizer or its apply_gradients method fails 
                self.nn.opt(gradient)  # use the neural network's optimizer directly with gradient value as input 
        except Exception as e:
            raise e  # raise any other exception 
        return output,loss  # return the output and loss values for this optimization step
    
    
    def pytorch_opt(self,data,labels):  # a method to perform one optimization step using pytorch platform
        output=self.nn.fp(data)  # get the output value using the neural network's forward propagation method
        loss=self.nn.loss(output,labels)  # get the loss value using the neural network's loss function
        try:
            try:
                self.nn.opt.zero_grad()  # use the neural network's optimizer's zero_grad method to clear the previous gradients 
                loss.backward()  # use pytorch's loss.backward method to calculate the gradients 
                self.nn.opt.step()  # use the neural network's optimizer's step method to update the neural network parameters with gradient value 
            except Exception:  # if the neural network does not have an optimizer or its zero_grad or step method fails 
                self.nn.opt(loss)  # use the neural network's optimizer directly with loss value as input 
        except Exception as e:
            raise e  # raise any other exception 
        return output,loss  # return the output and loss values for this optimization step
    
    
    def opt(self,data,labels):  # a method to perform one optimization step using either tensorflow or pytorch platform
        try:
            try:
                if self.platform.DType!=None:  # if tensorflow platform is used 
                    output,loss=self.tf_opt(data,labels)  # use the tf_opt method for optimization 
            except Exception:  # if tensorflow platform is not used or it fails 
                output,loss=self.pytorch_opt(data,labels)  # use the pytorch_opt method for optimization 
        except Exception as e:
            raise e  # raise any other exception 
        return output,loss  # return the output and loss values for this optimization step
    
    
    def _train(self,batch=None,test_batch=None):  # a method to perform one epoch of training
        if batch!=None:  # if batch mode is used
            total_loss=0  # initialize the total loss value for all batches
            total_acc=0  # initialize the total accuracy value for all batches
            if self.train_dataset!=None:  # if a tensorflow or pytorch dataset object is used for train data
                for data_batch,labels_batch in self.train_dataset:  # iterate over each batch of data and labels from the train dataset
                    if self.stop==True:  # if the stop flag is set to True
                        if self.stop_func():  # check whether the training has reached the target loss or accuracy values using the stop_func method
                            return  # return nothing and end the training
                    try:
                        data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)  # use the neural network's data_func method to preprocess the data and labels batch (optional)
                    except Exception as e:
                        try:
                            if self.nn.data_func!=None:  # if the neural network has a data_func method defined
                                raise e  # raise the exception
                        except Exception:
                            pass
                    output,batch_loss=self.opt(data_batch,labels_batch)  # perform one optimization step using the opt method and get the output and batch loss values
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc)  # calculate and update the total loss and accuracy values for all batches using the loss_acc method
            else:  # if a numpy array is used for train data
                total_loss=0  # initialize the total loss value for all batches
                total_acc=0  # initialize the total accuracy value for all batches
                batches=int((self.shape0-self.shape0%batch)/batch)  # calculate how many batches are needed for train data according to batch size
                for j in range(batches):  # iterate over each batch index
                    if self.stop==True:   # if the stop flag is set to True
                        if self.stop_func():   # check whether the training has reached the target loss or accuracy values using the stop_func method
                            return   # return nothing and end the training
                    index1=j*batch   # calculate the start index of train data for this batch
                    index2=(j+1)*batch   # calculate the end index of train data for this batch
                    data_batch,labels_batch=self.data_func(batch,index1,index2,j)   # get a batch of data and labels from train data and labels using the data_func method 
                    try:
                        data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch)   # use the neural network's data_func method to preprocess the data and labels batch (optional)
                    except Exception as e:
                        try:
                            if self.nn.data_func!=None:   # if the neural network has a data_func method defined
                                raise e   # raise the exception
                        except Exception:
                            pass
                    output,batch_loss=self.opt(data_batch,labels_batch)   # perform one optimization step                    
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc) # calculate and update the total loss and accuracy values for all batches using the loss_acc method
                    try:
                        try:
                            self.nn.bc.assign_add(1) # use the neural network's batch counter's assign_add method to increase the batch counter by 1 (for tensorflow platform)
                        except Exception: # if the neural network does not have a batch counter or its assign_add method fails
                            self.nn.bc+=1 # use the normal addition operation to increase the batch counter by 1 (for pytorch platform)
                    except Exception:
                        pass
                if self.shape0%batch!=0: # if there are some samples left in train data that are not enough to form a full batch
                    if self.stop==True: # if the stop flag is set to True
                        if self.stop_func(): # check whether the training has reached the target loss or accuracy values using the stop_func method
                            return # return nothing and end the training
                    batches+=1 # increase the number of batches by 1 to include the remaining samples
                    index1=batches*batch # calculate the start index of train data for the last batch
                    index2=batch-(self.shape0-batches*batch) # calculate how many samples are needed from the beginning of train data to form a full batch
                    data_batch,labels_batch=self.data_func(batch,index1,index2,flag=True) # get a batch of data and labels from train data and labels using the data_func method with flag set to True
                    try:
                        data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch) # use the neural network's data_func method to preprocess the data and labels batch (optional)
                    except Exception as e:
                        try:
                            if self.nn.data_func!=None: # if the neural network has a data_func method defined
                                raise e # raise the exception
                        except Exception:
                            pass
                    output,batch_loss=self.opt(data_batch,labels_batch) # perform one optimization step using the opt method and get the output and batch loss values
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc) # calculate and update the total loss and accuracy values for all batches using the loss_acc method
                    try:
                        try:
                            self.nn.bc.assign_add(1) # use the neural network's batch counter's assign_add method to increase the batch counter by 1 (for tensorflow platform)
                        except Exception: # if the neural network does not have a batch counter or its assign_add method fails
                            self.nn.bc+=1 # use the normal addition operation to increase the batch counter by 1 (for pytorch platform)
                    except Exception:
                        pass
            try:
                if self.platform.DType!=None: # if tensorflow platform is used 
                    loss=total_loss.numpy()/batches # convert the total loss value to numpy array and divide it by number of batches to get the average loss value for this epoch 
            except Exception: # if tensorflow platform is not used or it fails 
                loss=total_loss.detach().numpy()/batches # detach the total loss value from computation graph and convert it to numpy array and divide it by number of batches to get the average loss value for this epoch 
            try:
                train_acc=total_acc.numpy()/batches # convert the total accuracy value to numpy array and divide it by number of batches to get the average accuracy value for this epoch 
            except Exception as e: 
                try:
                    if self.nn.accuracy!=None: # if the neural network has an accuracy method defined 
                        raise e # raise the exception 
                except Exception:
                    pass
            self.train_loss=loss # assign the average loss value to train loss attribute 
            self.train_loss_list.append(loss) # append the average loss value to train loss list 
            try:
                self.train_acc=train_acc # assign the average accuracy value to train accuracy attribute 
                self.train_acc_list.append(train_acc) # append the average accuracy value to train accuracy list 
            except Exception as e: 
                try:
                    if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                        raise e  # raise the exception 
                except Exception:
                    pass
            if self.test_flag==True:  # if test data is used 
                self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)  # calculate the test loss and accuracy values using the test method 
                self.test_loss_list.append(self.test_loss)  # append the test loss value to test loss list 
                try:
                    self.test_acc_list.append(self.test_acc)  # append the test accuracy value to test accuracy list 
                except Exception as e: 
                    try:
                        if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                            raise e  # raise the exception 
                    except Exception:
                        pass
        else: # if batch mode is not used (use all data at once)
            output,train_loss=self.opt(self.train_data,self.train_labels) # perform one optimization step using the opt method and get the output and train loss values
            self.loss_acc(output=output,labels_batch=labels_batch,loss=train_loss,test_batch=test_batch,total_loss=total_loss,total_acc=total_acc) # calculate and update the train and test loss and accuracy values using the loss_acc method
        return # return nothing for this case
    
    
    def train(self,batch=None,epoch=None,test_batch=None,save=None,one=True,p=None,s=None): # a method to perform multiple epochs of training
        self.batch=batch # assign the batch size for training data to batch attribute
        self.epoch=0 # initialize the current epoch number to 0
        self.train_counter+=1 # increase the train counter by 1
        if p==None: # if p is None, it means the default value of p is used
            self.p=9 # assign 9 to p attribute, which means print the train and test loss and accuracy values every 10 epochs
        else:
            self.p=p-1 # assign p-1 to p attribute, which means print the train and test loss and accuracy values every p epochs
        if s==None: # if s is None, it means the default value of s is used
            self.s=1 # assign 1 to s attribute, which means save the model every epoch
            self.file_list=None # assign None to file_list attribute, which means do not keep track of saved files
        else:
            self.s=s-1 # assign s-1 to s attribute, which means save the model every s epochs
            self.file_list=[] # assign an empty list to file_list attribute, which means keep track of saved files
        if epoch!=None: # if epoch is not None, it means a fixed number of epochs is given
            for i in range(epoch): # iterate over each epoch index
                t1=time.time() # record the start time of this epoch
                self._train(batch,test_batch) # perform one epoch of training using the _train method
                if self.stop_flag==True: # if the stop flag is set to True, it means the training has reached the target loss or accuracy values and has been stopped
                    return # return nothing and end the training
                try:
                    try:
                        self.nn.ec.assign_add(1) # use the neural network's epoch counter's assign_add method to increase the epoch counter by 1 (for tensorflow platform)
                    except Exception: # if the neural network does not have an epoch counter or its assign_add method fails
                        self.nn.ec+=1 # use the normal addition operation to increase the epoch counter by 1 (for pytorch platform)
                except Exception:
                    pass
                self.total_epoch+=1 # increase the total epoch number by 1 for all trainings
                if epoch%10!=0: 
                    p=epoch-epoch%self.p 
                    p=int(p/self.p) 
                    s=epoch-epoch%self.s 
                    s=int(s/self.s) 
                else:
                    p=epoch/(self.p+1) 
                    p=int(p) 
                    s=epoch/(self.s+1) 
                    s=int(s) 
                if p==0: 
                    p=1 
                if s==0: 
                    s=1 
                if i%p==0:  # if this epoch index is a multiple of p (or p+1)
                    if self.test_flag==False:  # if test data is not used 
                        try:
                            if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))  # print the current epoch number and train loss value 
                                if self.acc_flag=='%':  # if percentage mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))  # print the current epoch number and train accuracy value in percentage 
                                else:  # if decimal mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))  # print the current epoch number and train accuracy value in                                    print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))  # print the current epoch number and train accuracy value in decimal
                                print() # print an empty line for readability
                        except Exception: # if the neural network does not have an accuracy method defined or it fails
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss)) # print the current epoch number and train loss value
                            print() # print an empty line for readability
                    else: # if test data is used
                        try:
                            if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))  # print the current epoch number, train loss value and test loss value 
                                if self.acc_flag=='%':  # if percentage mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))  # print the current epoch number, train accuracy value and test accuracy value in percentage 
                                else:  # if decimal mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.6f},test accuracy:{2:.6f}'.format(i+1,self.train_acc,self.test_acc))  # print the current epoch number, train accuracy value and test accuracy value in decimal
                                print() # print an empty line for readability
                        except Exception:  # if the neural network does not have an accuracy method defined or it fails
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))  # print the current epoch number, train loss value and test loss value 
                            print() # print an empty line for readability
                if save!=None and i%s==0:  # if save is not None, it means a file name is given to save the model, and if this epoch index is a multiple of s (or s+1)
                    self.save(self.total_epoch,one)  # save the model using the save method with total epoch number and one flag as inputs
                t2=time.time()  # record the end time of this epoch
                self.time+=(t2-t1)  # calculate and update the time elapsed for this training session
        else:  # if epoch is None, it means an infinite number of epochs is given
            i=0  # initialize the epoch index to 0
            while True:  # loop indefinitely until stopped by other conditions
                t1=time.time()  # record the start time of this epoch
                self._train(test_batch=test_batch)  # perform one epoch of training using the _train method
                if self.stop_flag==True:  # if the stop flag is set to True, it means the training has reached the target loss or accuracy values and has been stopped
                    return  # return nothing and end the training
                i+=1  # increase the epoch index by 1
                try:
                    try:
                        self.nn.ec.assign_add(1)  # use the neural network's epoch counter's assign_add method to increase the epoch counter by 1 (for tensorflow platform)
                    except Exception:  # if the neural network does not have an epoch counter or its assign_add method fails
                        self.nn.ec+=1  # use the normal addition operation to increase the epoch counter by 1 (for pytorch platform)
                except Exception:
                    pass
                self.total_epoch+=1  # increase the total epoch number by 1 for all trainings
                if epoch%10!=0: 
                    p=epoch-epoch%self.p 
                    p=int(p/self.p) 
                    s=epoch-epoch%self.s 
                    s=int(s/self.s) 
                else:
                    p=epoch/(self.p+1) 
                    p=int(p) 
                    s=epoch/(self.s+1) 
                    s=int(s) 
                if p==0: 
                    p=1 
                if s==0: 
                    s=1 
                if i%p==0:   # if this epoch index is a multiple of p (or p+1)
                    if self.test_flag==False:   # if test data is not used 
                        try:
                            if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined 
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))   # print the current epoch number and train loss value 
                                if self.acc_flag=='%':   # if percentage mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))   # print the current epoch number and train accuracy value in percentage 
                                else:   # if decimal mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))   # print the current epoch number and train accuracy value in decimal
                                print()  # print an empty line for readability
                        except Exception:  # if the neural network does not have an accuracy method defined or it fails
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))  # print the current epoch number and train loss value
                            print()  # print an empty line for readability
                    else:  # if test data is used
                        try:
                            if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined 
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))   # print the current epoch number, train loss value and test loss value 
                                if self.acc_flag=='%':   # if percentage mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))   # print the current epoch number, train accuracy value and test accuracy value in percentage 
                                else:   # if decimal mode is used for accuracy display 
                                    print('epoch:{0}   accuracy:{1:.6f},test accuracy:{2:.6f}'.format(i+1,self.train_acc,self.test_acc))   # print the current epoch number, train accuracy value and test accuracy value in decimal
                                print()  # print an empty line for readability
                        except Exception:   # if the neural network does not have an accuracy method defined or it fails
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))   # print the current epoch number, train loss value and test loss value 
                            print()  # print an empty line for readability
                if save!=None and i%s==0:   # if save is not None, it means a file name is given to save the model, and if this epoch index is a multiple of s (or s+1)
                    self.save(self.total_epoch,one)   # save the model using the save method with total epoch number and one flag as inputs
                t2=time.time()   # record the end time of this epoch
                self.time+=(t2-t1)   # calculate and update the time elapsed for this training session
        if save!=None:  # if save is not None, it means a file name is given to save the model
            self.save()  # save the model using the save method without any inputs (use default values)
        self._time=self.time-int(self.time)  # get the fractional part of the time elapsed for this training session
        if self._time<0.5:  # if the fractional part is less than 0.5
            self.time=int(self.time)  # round down the time elapsed to integer
        else:  # if the fractional part is greater than or equal to 0.5
            self.time=int(self.time)+1  # round up the time elapsed to integer
        self.total_time+=self.time  # calculate and update the total time elapsed for all trainings
        if self.test_flag==False:  # if test data is not used
            print('last loss:{0:.6f}'.format(self.train_loss))  # print the last train loss value
        else:  # if test data is used
            print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))  # print the last train loss value and test loss value
        try:
            if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                if self.acc_flag=='%':  # if percentage mode is used for accuracy display 
                    if self.test_flag==False:  # if test data is not used 
                        print('last accuracy:{0:.1f}'.format(self.train_acc*100))  # print the last train accuracy value in percentage 
                    else:  # if test                    else:  # if test data is used 
                        print('last accuracy:{0:.1f},last test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))  # print the last train accuracy value and test accuracy value in percentage 
                else:  # if decimal mode is used for accuracy display 
                    if self.test_flag==False:  # if test data is not used 
                        print('last accuracy:{0:.6f}'.format(self.train_acc))  # print the last train accuracy value in decimal 
                    else:  # if test data is used 
                        print('last accuracy:{0:.6f},last test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))  # print the last train accuracy value and test accuracy value in decimal
        except Exception:  # if the neural network does not have an accuracy method defined or it fails
            pass
        print()  # print an empty line for readability
        print('time:{0}s'.format(self.time))  # print the time elapsed for this training session
        self.training_flag=False  # set the training flag to False, which means the training is finished
        return  # return nothing and end the training
    
    
    def train_ol(self):  # a method to perform online learning, which means updating the model with one sample at a time
        while True:  # loop indefinitely until stopped by other conditions
            if self.stop_flag==True:  # if the stop flag is set to True, it means the online learning has reached the target loss or accuracy values and has been stopped
                return  # return nothing and end the online learning
            if self.save_flag==True:  # if the save flag is set to True, it means a request to save the model has been made
                self.save()  # save the model using the save method without any inputs (use default values)
            self.suspend_func()  # check whether a request to suspend the online learning has been made using the suspend_func method
            data=self.nn.ol()  # get one sample of data and label from the neural network's ol method, which should be defined by the user
            if data=='stop':  # if the data is 'stop', it means a request to stop the online learning has been made
                return  # return nothing and end the online learning
            elif data=='suspend':  # if the data is 'suspend', it means a request to suspend the online learning has been made
                self.nn.suspend=True  # set the neural network's suspend attribute to True, which means the online learning is suspended
                while True:  # loop indefinitely until resumed by other conditions
                    if self.nn.suspend==False:  # if the neural network's suspend attribute is set to False, it means a request to resume the online learning has been made
                        break  # break the loop and resume the online learning
                continue  # continue to the next iteration of online learning
            output,loss=self.opt(data[0],data[1])  # perform one optimization step using the opt method and get the output and loss values
            loss=loss.numpy()  # convert the loss value to numpy array
            if len(self.nn.train_loss_list)==self.nn.max_length:  # if the train loss list has reached its maximum length, which should be defined by the user
                del self.nn.train_loss_list[0]  # delete the first element of train loss list
            self.nn.train_loss_list.append(loss)  # append the loss value to train loss list
            try:
                train_acc=self.nn.accuracy(output,data[1])  # calculate the accuracy value using the neural network's accuracy method
                if len(self.nn.train_acc_list)==self.nn.max_length:  # if the train accuracy list has reached its maximum length, which should be defined by the user
                    del self.nn.train_acc_list[0]  # delete the first element of train accuracy list
                self.train_acc_list.append(train_acc)  # append the accuracy value to train accuracy list
            except Exception as e:
                try:
                    if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                        raise e  # raise the exception 
                except Exception:
                    pass
            try:
                self.nn.c+=1  # increase the neural network's counter by 1, which should be defined by the user to keep track of how many samples have been used for online learning 
            except Exception:
                pass
        return   # return nothing and end the online learning
    
    
    def test(self,test_data=None,test_labels=None,batch=None):  # a method to calculate the test loss and accuracy values using test data and labels
        if batch!=None:  # if batch mode is used
            total_loss=0  # initialize the total loss value for all batches
            total_acc=0  # initialize the total accuracy value for all batches
            if self.test_dataset!=None:  # if a tensorflow or pytorch dataset object is used for test data
                for data_batch,labels_batch in self.test_dataset:  # iterate over each batch of data and labels from the test dataset
                    output=self.nn.fp(data_batch)  # get the output value using the neural network's forward propagation method
                    batch_loss=self.nn.loss(output,labels_batch)  # get the batch loss value using the neural network's loss function
                    total_loss+=batch_loss  # accumulate the batch loss to total loss
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch)  # calculate the batch accuracy using the neural network's accuracy method
                        total_acc+=batch_acc  # accumulate the batch accuracy to total accuracy
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                                raise e  # raise the exception 
                        except Exception:
                            pass
            else:  # if a numpy array is used for test data
                total_loss=0  # initialize the total loss value for all batches
                total_acc=0  # initialize the total accuracy value for all batches
                if type(test_data)==list:  # if test data is a list of arrays (for multiple inputs)
                    batches=int((test_data[0].shape[0]-test_data[0].shape[0]%batch)/batch)  # calculate how many batches are needed for test data according to batch size
                    shape0=test_data[0].shape[0]  # get the number of samples in test data
                else:  # if test data is a single array 
                    batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)  # calculate how many batches are needed for test data according to batch size
                    shape0=test_data.shape[0]  # get the number of samples in test data
                for j in range(batches):  # iterate over each batch index
                    index1=j*batch  # calculate the start index of test data for this batch
                    index2=(j+1)*batch  # calculate the end index of test data for this batch
                    if type(test_data)==list:  # if test data is a list of arrays (for multiple inputs)
                        for i in range(len(test_data)):  
                            data_batch[i]=test_data[i][index1:index2]  # get a slice of test data according to the index range for each input array 
                    else:  # if test data is a single array 
                        data_batch=test_data[index1:index2]  # get a slice of test data according to the index range 
                    if type(test_labels)==list:  # if test labels is a list of arrays (for multiple outputs)
                        for i in range(len(test_labels)):  
                            labels_batch[i]=test_labels[i][index1:index2]  # get a slice of test labels according to the index range for each output array 
                    else:  # if test labels is a single array 
                        labels_batch=test_labels[index1:index2]  # get a slice of test labels according to the index range 
                    output=self.nn.fp(data_batch)   # get the output value using the neural network's forward propagation method 
                    batch_loss=self.nn.loss(output,labels_batch)   # get the batch loss value using the neural network's loss function 
                    total_loss+=batch_loss   # accumulate the batch loss to total loss 
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch)   # calculate the batch accuracy using the neural network's accuracy method 
                        total_acc+=batch_acc   # accumulate the batch accuracy to total accuracy 
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined 
                                raise e   # raise the exception 
                        except Exception:
                            pass
                if shape0%batch!=0:   # if there are some samples left in test data that are not enough to form a full batch 
                    batches+=1   # increase the number of batches by 1 to include the remaining samples 
                    index1=batches*batch   # calculate the start index of test data for the last batch 
                    index2=batch-(shape0-batches*batch)   # calculate how many samples are needed from the beginning of test data to form a full batch 
                    try:
                        try:
                            if type(test_data)==list:   # if test data is a list of arrays (for multiple                            if type(test_data)==list:   # if test data is a list of arrays (for multiple inputs)
                                for i in range(len(test_data)):  
                                    data_batch[i]=self.platform.concat([test_data[i][index1:],test_data[i][:index2]],0)  # concatenate two slices of test data from the end and the beginning to form a batch for each input array 
                            else:  # if test data is a single array 
                                data_batch=self.platform.concat([test_data[index1:],test_data[:index2]],0)  # concatenate two slices of test data from the end and the beginning to form a batch 
                            if type(test_labels)==list:  # if test labels is a list of arrays (for multiple outputs)
                                for i in range(len(test_labels)):  
                                    labels_batch[i]=self.platform.concat([test_labels[i][index1:],test_labels[i][:index2]],0)  # concatenate two slices of test labels from the end and the beginning to form a batch for each output array 
                            else:  # if test labels is a single array 
                                labels_batch=self.platform.concat([test_labels[index1:],test_labels[:index2]],0)  # concatenate two slices of test labels from the end and the beginning to form a batch 
                        except Exception:  # if the platform's concat method fails 
                            if type(test_data)==list:  # if test data is a list of arrays (for multiple inputs)
                                for i in range(len(test_data)):  
                                    data_batch[i]=np.concatenate([test_data[i][index1:],test_data[i][:index2]],0)  # use numpy's concatenate method instead for each input array 
                            else:  # if test data is a single array 
                                data_batch=np.concatenate([test_data[index1:],test_data[:index2]],0)  # use numpy's concatenate method instead 
                            if type(test_labels)==list:  # if test labels is a list of arrays (for multiple outputs)
                                for i in range(len(test_labels)):  
                                    labels_batch[i]=np.concatenate([test_labels[i][index1:],test_labels[i][:index2]],0)  # use numpy's concatenate method instead for each output array 
                            else:  # if test labels is a single array 
                                labels_batch=np.concatenate([test_labels[index1:],test_labels[:index2]],0)  # use numpy's concatenate method instead 
                    except Exception as e:
                        raise e  # raise any other exception 
                    output=self.nn.fp(data_batch)   # get the output value using the neural network's forward propagation method 
                    batch_loss=self.nn.loss(output,labels_batch)   # get the batch loss value using the neural network's loss function 
                    total_loss+=batch_loss   # accumulate the batch loss to total loss 
                    try:
                        batch_acc=self.nn.accuracy(output,labels_batch)   # calculate the batch accuracy using the neural network's accuracy method 
                        total_acc+=batch_acc   # accumulate the batch accuracy to total accuracy 
                    except Exception as e:
                        try:
                            if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined 
                                raise e   # raise the exception 
                        except Exception:
                            pass
            test_loss=total_loss.numpy()/batches   # convert the total loss value to numpy array and divide it by number of batches to get the average loss value for test data 
            try:
                if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined 
                    test_acc=total_acc.numpy()/batches   # convert the total accuracy value to numpy array and divide it by number of batches to get the average accuracy value for test data 
            except Exception:
                pass
        else:  # if batch mode is not used (use all data at once)
            output=self.nn.fp(test_data)   # get the output value using the neural network's forward propagation method 
            test_loss=self.nn.loss(output,test_labels)   # get the loss value using the neural network's loss function
            test_loss=test_loss.numpy()   # convert the loss value to numpy array
            try:
                test_acc=self.nn.accuracy(output,test_labels)   # calculate the accuracy value using the neural network's accuracy method
                test_acc=test_acc.numpy()   # convert the accuracy value to numpy array
            except Exception as e:
                try:
                    if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined
                        raise e   # raise the exception
                except Exception:
                    pass
        try:
            if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined
                return test_loss,test_acc   # return the test loss and accuracy values
        except Exception:
            return test_loss,None   # return the test loss value and None
    
    
    def suspend_func(self):  # a method to check whether a request to suspend the training or online learning has been made
        if self.suspend==True:  # if the suspend flag is set to True
            if self.save_epoch==None:  # if the save epoch number is None, it means no request to save the model has been made
                print('Training have suspended.')  # print a message to indicate the training or online learning has been suspended
            else:
                self._save()  # save the model using the _save method
            while True:  # loop indefinitely until resumed by other conditions
                if self.suspend==False:  # if the suspend flag is set to False, it means a request to resume the training or online learning has been made
                    print('Training have continued.')  # print a message to indicate the training or online learning has been resumed
                    break  # break the loop and resume the training or online learning
        return  # return nothing for this case
    
    
    def stop_func(self):  # a method to check whether the training or online learning has reached the target loss or accuracy values and stop it accordingly
        if self.end():  # check whether the target loss or accuracy values have been reached using the end method
            self.save(self.total_epoch,True)  # save the model using the save method with total epoch number and True flag as inputs
            print('\nSystem have stopped training,Neural network have been saved.')  # print a message to indicate the training or online learning has been stopped and the model has been saved
            self._time=self.time-int(self.time)  # get the fractional part of the time elapsed for this training session
            if self._time<0.5:  # if the fractional part is less than 0.5
                self.time=int(self.time)  # round down the time elapsed to integer
            else:  # if the fractional part is greater than or equal to 0.5
                self.time=int(self.time)+1  # round up the time elapsed to integer
            self.total_time+=self.time  # calculate and update the total time elapsed for all trainings
            print()  # print an empty line for readability
            print('epoch:{0}'.format(self.total_epoch))  # print the total epoch number for all trainings
            if self.test_flag==False:  # if test data is not used 
                print('last loss:{0:.6f}'.format(self.train_loss))  # print the last train loss value 
            else:  # if test data is used 
                print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))  # print the last train loss value and test loss value 
            try:
                if self.nn.accuracy!=None:   # if the neural network has an accuracy method defined 
                    if self.acc_flag=='%':   # if percentage mode is used for accuracy display 
                        if self.test_flag==False:   # if test data is not used 
                            print('last accuracy:{0:.1f}'.format(self.train_acc*100))   # print the last train accuracy value in percentage 
                        else:   # if test data is used 
                            print('last accuracy:{0:.1f},last test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))   # print the last train accuracy value and test accuracy value in percentage 
                    else:   # if decimal mode is used for accuracy display 
                        if self.test_flag==False:   # if test data is not used 
                            print('last accuracy:{0:.6f}'.format(self.train_acc))   # print the last train accuracy value in decimal 
                        else:   # if test data is used 
                            print('last accuracy:{0:.6f},last test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))   # print the last train accuracy value and test accuracy value in decimal 
            except Exception:   # if the neural network does not have an accuracy method defined or it fails 
                pass
            print()  # print an empty line for readability
            print('time:{0}s'.format(self.total_time))  # print the total time elapsed for all trainings
            self.stop_flag=True  # set the stop flag to True, which means the training or online learning has been stopped
            return True  # return True to indicate that the training or online learning has been stopped
        return False  # return False to indicate that the training or online learning has not been stopped
    
    
    def stop_func_(self):  # a method to check whether a request to stop the training or online learning has been made or the target loss or accuracy values have been reached
            if self.stop==True:  # if the stop flag is set to True, it means a request to stop the training or online learning has been made
                if self.stop_flag==True or self.stop_func():  # if the stop flag is already True, it means the target loss or accuracy values have been reached, or check whether the target loss or accuracy values have been reached using the stop_func method
                    return True  # return True to indicate that the training or online learning has been stopped
            return False  # return False to indicate that the training or online learning has not been stopped
    
    
    def visualize_train(self):  # a method to visualize the train loss and accuracy values using matplotlib.pyplot
        print()  # print an empty line for readability
        plt.figure(1)  # create a new figure with number 1
        plt.plot(np.arange(self.total_epoch),self.train_loss_list)  # plot the train loss list with x-axis as the epoch number
        plt.title('train loss')  # set the title of the plot to 'train loss'
        plt.xlabel('epoch')  # set the x-axis label to 'epoch'
        plt.ylabel('loss')  # set the y-axis label to 'loss'
        print('train loss:{0:.6f}'.format(self.train_loss))  # print the last train loss value
        try:
            if self.nn.accuracy!=None:  # if the neural network has an accuracy method defined 
                plt.figure(2)  # create a new figure with number 2
                plt.plot(np.arange(self.total_epoch),self.train_acc_list)  # plot the train accuracy list with x-axis as the epoch number
                plt.title('train acc')  # set the title of the plot to 'train acc'
                plt.xlabel('epoch')  # set the x-axis label to 'epoch'
                plt.ylabel('acc')  # set the y-axis label to 'acc'
                if self.acc_flag=='%':  # if percentage mode is used for accuracy display 
                    print('train acc:{0:.1f}'.format(self.train_acc*100))  # print the last train accuracy value in percentage 
                else:  # if decimal mode is used for accuracy display 
                    print('train acc:{0:.6f}'.format(self.train_acc))   # print the last train accuracy value in decimal 
        except Exception:  # if the neural network does not have an accuracy method defined or it fails 
            pass
        return   # return nothing for this case