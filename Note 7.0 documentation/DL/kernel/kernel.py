from tensorflow import function # import the function decorator from tensorflow
from multiprocessing import Process # import the Process class from multiprocessing
import numpy as np # import numpy as np
from Note.DL.dl.test import parallel_test # import the parallel_test class from Note.DL.dl.test
import time # import time


class kernel: # define a class
    def __init__(self,nn=None): 
        # define the constructor
        self.nn=nn # assign the nn argument to the self.nn attribute
        if hasattr(self.nn,'km'): # check if the nn object has the km attribute
            self.nn.km=1 # set the km attribute to 1
        self.platform=None # initialize the platform attribute to None
        self.batches=None # initialize the batches attribute to None
        self.process_t=None # initialize the process_t attribute to None
        self.prefetch_batch_size_t=None # initialize the prefetch_batch_size_t attribute to None
        self.suspend=False # initialize the suspend attribute to False
        self.stop=False # initialize the stop attribute to False
        self.stop_flag=False # initialize the stop_flag attribute to False
        self.save_epoch=None # initialize the save_epoch attribute to None
        self.batch=None # initialize the batch attribute to None
        self.epoch=0 # initialize the epoch attribute to 0
        self.end_loss=None # initialize the end_loss attribute to None
        self.end_acc=None # initialize the end_acc attribute to None
        self.end_test_loss=None # initialize the end_test_loss attribute to None
        self.end_test_acc=None # initialize the end_test_acc attribute to None
        self.acc_flag='%' # initialize the acc_flag attribute to '%'
        self.train_counter=0 # initialize the train_counter attribute to 0
        self.filename='save.dat' # initialize the filename attribute to 'save.dat'
        self.train_loss=None # initialize the train_loss attribute to None
        self.train_acc=None # initialize the train_acc attribute to None
        self.train_loss_list=[] # initialize the train_loss_list attribute to an empty list
        self.train_acc_list=[] # initialize the train_acc_list attribute to an empty list
        self.test_loss=None # initialize the test_loss attribute to None
        self.test_acc=None # initialize the test_acc attribute to None
        self.test_loss_list=[] # initialize the test_loss_list attribute to an empty list
        self.test_acc_list=[] # initialize the test_acc_list attribute to an empty list
        self.test_flag=False # initialize the test_flag attribute to False
        self.total_epoch=0 # initialize the total_epoch attribute to 0
        self.time=0 # initialize the time attribute to 0
        self.total_time=0 # initialize the total_time attribute to 0
    
    
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None): 
        # define a method for setting up data attributes
        if train_data is not None and type(self.nn.param[0])!=list: # check if train_data is not None and the type of the first element of self.nn.param is not list
            self.train_data=train_data.astype(self.nn.param[0].dtype.name) # convert the train_data to the same data type as the first element of self.nn.param and assign it to self.train_data
            self.train_labels=train_labels.astype(self.nn.param[0].dtype.name) # convert the train_labels to the same data type as the first element of self.nn.param and assign it to self.train_labels
        elif train_data is not None: # check if train_data is not None
            self.train_data=train_data.astype(self.nn.param[0][0].dtype.name) # convert the train_data to the same data type as the first element of the first element of self.nn.param and assign it to self.train_data
            self.train_labels=train_labels.astype(self.nn.param[0][0].dtype.name) # convert the train_labels to the same data type as the first element of the first element of self.nn.param and assign it to self.train_labels
        self.train_dataset=train_dataset # assign the train_dataset argument to the self.train_dataset attribute
        self.test_data=test_data # assign the test_data argument to the self.test_data attribute
        self.test_labels=test_labels # assign the test_labels argument to the self.test_labels attribute
        self.test_dataset=test_dataset # assign the test_dataset argument to the self.test_dataset attribute
        if test_data is not None or test_dataset is not None: # check if test_data or test_dataset is not None
            self.test_flag=True # set the test_flag attribute to True
        if train_data is not None: # check if train_data is not None
            self.shape0=train_data.shape[0] # get the first dimension of train_data and assign it to self.shape0
        return # return from the method
    
    
    def init(self): 
        # define a method for initializing some attributes
        self.suspend=False # set the suspend attribute to False
        self.stop=False # set the stop attribute to False
        self.stop_flag=False # set the stop_flag attribute to False
        self.save_epoch=None # set the save_epoch attribute to None
        self.end_loss=None # set the end_loss attribute to None
        self.end_acc=None # set the end_acc attribute to None
        self.end_test_loss=None # set the end_test_loss attribute to None
        self.end_test_acc=None # set the end_test_acc attribute to None
        self.train_loss=None # set the train_loss attribute to None
        self.train_acc=None # set the train_acc attribute to None
        self.test_loss=None # set the test_loss attribute to None
        self.test_acc=None # set the test_acc attribute to None
        self.train_loss_list.clear() # clear the train_loss_list attribute
        self.train_acc_list.clear() # clear the train_acc_list attribute
        self.test_loss_list.clear() # clear the test_loss_list attribute
        self.test_acc_list.clear() # clear the test_acc_list attribute
        self.test_flag=False # set the test_flag attribute to False
        self.train_counter=0 # set the train_counter attribute to 0
        self.epoch=0 # set the epoch attribute to 0
        self.total_epoch=0 # set the total_epoch attribute to 0
        self.time=0 # set the time attribute to 0
        self.total_time=0 # set the total_time attribute to 0
        return # return from the method
    
    
    def end(self):
        # define a method for checking if some conditions are met for ending training
        if self.end_acc!=None and self.train_acc!=None and self.train_acc>self.end_acc:
            return True
        elif self.end_loss!=None and self.train_loss!=None and self.train_loss<self.end_loss:
            return True
        elif self.end_test_acc!=None and self.test_acc!=None and self.test_acc>self.end_test_acc:
            return True
        elif self.end_test_loss!=None and self.test_loss!=None and self.test_loss<self.end_test_loss:
            return True
        elif self.end_acc!=None and self.end_test_acc!=None:
            if self.train_acc!=None and self.test_acc!=None and self.train_acc>self.end_acc and self.test_acc>self.end_test_acc:
                return True
        elif self.end_loss!=None and self.end_test_loss!=None:
            if self.train_loss!=None and self.test_loss!=None and self.train_loss<self.end_loss and self.test_loss<self.end_test_loss:
                return True
    
    
    def loss_acc(self,output=None,labels_batch=None,loss=None,test_batch=None,total_loss=None,total_acc=None): 
        # define a method for calculating loss and accuracy for each batch
        if self.batch!=None: # check if the batch attribute is not None
            total_loss+=loss # add loss to total_loss
            if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                batch_acc=self.nn.accuracy(output,labels_batch) # call accuracy method with output and labels_batch as arguments and assign it to batch_acc
                total_acc+=batch_acc # add batch_acc to total_acc
            return total_loss,total_acc # return total_loss and total_acc
        else: # check if the batch attribute is None
            loss=loss.numpy() # convert loss to numpy array and assign it to loss
            self.train_loss=loss # assign loss to self.train_loss
            self.train_loss_list.append(loss) # append loss to self.train_loss_list
            if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                acc=self.nn.accuracy(output,self.train_labels) # call accuracy method with output and self.train_labels as arguments and assign it to acc
                acc=acc.numpy() # convert acc to numpy array and assign it to acc
                self.train_acc=acc # assign acc to self.train_acc
                self.train_acc_list.append(acc) # append acc to self.train_acc_list
            if self.test_flag==True: # check if the test_flag attribute is True
                if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch) # call test method with self.test_data, self.test_labels, and test_batch as arguments and assign the results to self.test_loss and self.test_acc
                    self.test_loss_list.append(self.test_loss) # append self.test_loss to self.test_loss_list
                    self.test_acc_list.append(self.test_acc) # append self.test_acc to self.test_acc_list
                else: # check if the nn object does not have accuracy method
                    self.test_loss=self.test(self.test_data,self.test_labels,test_batch) # call test method with self.test_data, self.test_labels, and test_batch as arguments and assign the result to self.test_loss
                    self.test_loss_list.append(self.test_loss) # append self.test_loss to self.test_loss_list
            return # return from the method
    
    
    def data_func(self,batch=None,index1=None,index2=None,j=None,flag=None): 
        # define a method for getting a batch of data and labels from train_data and train_labels attributes
        if flag==None: # check if flag is None
            if batch!=1: # check if batch is not 1
                data_batch=self.train_data[index1:index2] # slice the train_data attribute from index1 to index2 and assign it to data_batch
            else: # check if batch is 1
                data_batch=self.train_data[j] # get the j-th element of train_data attribute and assign it to data_batch
            if batch!=1: # check if batch is not 1
                labels_batch=self.train_labels[index1:index2] # slice the train_labels attribute from index1 to index2 and assign it to labels_batch
            else: # check if batch is 1
                labels_batch=self.train_labels[j] # get the j-th element of train_labels attribute and assign it to labels_batch
        else: # check if flag is not None
            try: # try the following block of code
                try: # try the following block of code
                    data_batch=self.platform.concat([self.train_data[index1:],self.train_data[:index2]],0) 
                    # concatenate the train_data attribute from index1 to the end and from the beginning to index2 along the first axis and assign it to data_batch 
                    labels_batch=self.platform.concat([self.train_labels[index1:],self.train_labels[:index2]],0) 
                    # concatenate the train_labels attribute from index1 to the end and from the beginning to index2 along the first axis and assign it to labels_batch 
                except Exception: # handle any exception raised in the previous block of code
                    data_batch=np.concatenate([self.train_data[index1:],self.train_data[:index2]],0) 
                    # concatenate the train_data attribute from index1 to the end and from the beginning to index2 along the first axis using numpy and assign it to data_batch 
                    labels_batch=np.concatenate([self.train_labels[index1:],self.train_labels[:index2]],0) 
                    # concatenate the train_labels attribute from index1 to the end and from the beginning to index2 along the first axis using numpy and assign it to labels_batch 
            except Exception as e: # handle any exception raised in the previous block of code and assign it to e
                raise e # re-raise e
        return data_batch,labels_batch # return data_batch and labels_batch
    
    
    @function(jit_compile=True) 
    # use function decorator with jit_compile argument set to True for just-in-time compilation of this method for faster execution 
    def tf_opt(self,data,labels): 
        # define a method for optimizing parameters using tensorflow platform
        try: # try the following block of code
            try: # try the following block of code
                with self.platform.GradientTape(persistent=True) as tape: # create a GradientTape object with persistent argument set to True and assign it to tape
                    try: # try the following block of code
                        output=self.nn.fp(data) # call fp method of the nn object with data as argument and assign it to output
                        loss=self.nn.loss(output,labels) # call loss method of the nn object with output and labels as arguments and assign it to loss
                    except Exception: # handle any exception raised in the previous block of code
                        output,loss=self.nn.fp(data,labels) # call fp method of the nn object with data and labels as arguments and assign the results to output and loss
            except Exception: # handle any exception raised in the previous block of code
                if hasattr(self.nn,'GradientTape'): # check if the nn object has GradientTape method
                    tape,output,loss=self.nn.GradientTape(data,labels) # call GradientTape method of the nn object with data and labels as arguments and assign the results to tape, output, and loss
        except Exception as e: # handle any exception raised in the previous block of code and assign it to e
            raise e # re-raise e
        if hasattr(self.nn,'gradient'): # check if the nn object has gradient method
            gradient=self.nn.gradient(tape,loss) # call gradient method of the nn object with tape and loss as arguments and assign it to gradient
        else: # check if the nn object does not have gradient method
            gradient=tape.gradient(loss,self.nn.param) # call gradient method of the tape object with loss and self.nn.param as arguments and assign it to gradient
        if hasattr(self.nn.opt,'apply_gradients'): # check if the opt attribute of the nn object has apply_gradients method
            self.nn.opt.apply_gradients(zip(gradient,self.nn.param)) # call apply_gradients method of the opt attribute of the nn object with zip object of gradient and self.nn.param as argument
        else: # check if the opt attribute of the nn object does not have apply_gradients method
            self.nn.opt(gradient) # call opt attribute of the nn object with gradient as argument
        return output,loss # return output and loss
    
    
    def pytorch_opt(self,data,labels): 
        # define a method for optimizing parameters using pytorch platform
        output=self.nn.fp(data) # call fp method of the nn object with data as argument and assign it to output
        loss=self.nn.loss(output,labels) # call loss method of the nn object with output and labels as arguments and assign it to loss
        if hasattr(self.nn.opt,'zero_grad'): # check if the opt attribute of the nn object has zero_grad method
            self.nn.opt.zero_grad() # call zero_grad method of the opt attribute of the nn object
            loss.backward() # call backward method of the loss object
            self.nn.opt.step() # call step method of the opt attribute of the nn object
        else: # check if the opt attribute of the nn object does not have zero_grad method
            self.nn.opt(loss) # call opt attribute of the nn object with loss as argument
        return output,loss # return output and loss
    
    
    def opt(self,data,labels): 
        # define a method for optimizing parameters using different platforms
        if hasattr(self.platform,'DType'): 
        # check if the platform attribute has DType attribute, which indicates tensorflow platform 
            output,loss=self.tf_opt(data,labels) 
            # call tf_opt method with data and labels as arguments and assign the results to output and loss 
        else: 
        # check if the platform attribute does not have DType attribute, which indicates pytorch platform 
            output,loss=self.pytorch_opt(data,labels) 
            # call pytorch_opt method with data and labels as arguments and assign the results to output and loss 
        return output,loss 
        # return output and loss
    
    
    def _train(self,batch=None,test_batch=None): 
        # define a private method for training on one epoch or iteration 
        if batch!=None: # check if batch is not None
            total_loss=0 # initialize total_loss to 0
            total_acc=0 # initialize total_acc to 0
            if self.train_dataset!=None: # check if train_dataset attribute is not None
                for data_batch,labels_batch in self.train_dataset: # iterate over the train_dataset attribute
                    if self.stop==True: # check if stop attribute is True
                        if self.stop_func(): # call stop_func method and check if it returns True
                            return # return from the method
                    if hasattr(self.nn,'data_func'): # check if the nn object has data_func method
                        data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch) # call data_func method of the nn object with data_batch and labels_batch as arguments and assign the results to data_batch and labels_batch
                    output,batch_loss=self.opt(data_batch,labels_batch) # call opt method with data_batch and labels_batch as arguments and assign the results to output and batch_loss
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc) 
                    # call loss_acc method with output, labels_batch, batch_loss, total_loss, and total_acc as arguments and assign the results to total_loss and total_acc 
                    if hasattr(self.nn,'bc'): # check if the nn object has bc attribute
                        try: # try the following block of code
                            self.nn.bc.assign_add(1) # call assign_add method of the bc attribute of the nn object with 1 as argument
                        except Exception: # handle any exception raised in the previous block of code
                            self.nn.bc+=1 # add 1 to the bc attribute of the nn object
                else: # check if train_dataset attribute is None
                    shape0=self.train_data.shape[0] # get the first dimension of train_data attribute and assign it to shape0
                    batches=int((shape0-shape0%batch)/batch) # calculate the number of batches and assign it to batches
                    for j in range(batches): # iterate over batches
                        if self.stop==True: # check if stop attribute is True
                            if self.stop_func(): # call stop_func method and check if it returns True
                                return # return from the method
                        index1=j*batch # calculate the start index of a batch and assign it to index1
                        index2=(j+1)*batch # calculate the end index of a batch and assign it to index2
                        data_batch,labels_batch=self.data_func(batch,index1,index2,j) 
                        # call data_func method with batch, index1, index2, and j as arguments and assign the results to data_batch and labels_batch 
                        if hasattr(self.nn,'data_func'): 
                        # check if the nn object has data_func method 
                            data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch) 
                            # call data_func method of the nn object with data_batch and labels_batch as arguments and assign the results to data_batch and labels_batch 
                        output,batch_loss=self.opt(data_batch,labels_batch) 
                        # call opt method with data_batch and labels_batch as arguments and assign the results to output and batch_loss 
                        total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc) 
                        # call loss_acc method with output, labels_batch, batch_loss, total_loss, and total_acc as arguments and assign the results to total_loss and total_acc 
                        if hasattr(self.nn,'bc'): 
                        # check if the nn object has bc attribute 
                            try: 
                            # try the following block of code 
                                self.nn.bc.assign_add(1) 
                                # call assign_add method of the bc attribute of the nn object with 1 as argument 
                            except Exception: 
                            # handle any exception raised in the previous block of code 
                                self.nn.bc+=1 
                                # add 1 to the bc attribute of the nn object 
                    if shape0%batch!=0: 
                    # check if there is a remainder after dividing shape0 by batch 
                        batches+=1 
                        # add 1 to batches 
                        index1=batches*batch 
                        # calculate the start index of a batch and assign it to index1 
                        index2=batch-(shape0-batches*batch) 
                        # calculate the end index of a batch and assign it to index2 
                        try: 
                        # try the following block of code 
                            data_batch,labels_batch=self.data_func(batch,index1,index2,flag=True) 
                            # call data_func method with batch, index1, index2, and flag set to True as arguments and assign the results to data_batch and labels_batch 
                            if hasattr(self.nn,'data_func'): 
                            # check if the nn object has data_func method 
                                data_batch,labels_batch=self.nn.data_func(data_batch,labels_batch) 
                                # call data_func method of the nn object with data_batch and labels_batch as arguments and assign the results to data_batch and labels_batch 
                            output,batch_loss=self.opt(data_batch,labels_batch) 
                            # call opt method with data_batch and labels_batch as arguments and assign the results to output and batch_loss 
                            total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc) 
                            # call loss_acc method with output, labels_batch, batch_loss, total_loss, and total_acc as arguments and assign the results to total_loss and total_acc 
                            if hasattr(self.nn,'bc'): 
                            # check if the nn object has bc attribute 
                                try: 
                                # try the following block of code 
                                    self.nn.bc.assign_add(1) 
                                    # call assign_add method of the bc attribute of the nn object with 1 as argument 
                                except Exception: 
                                # handle any exception raised in the previous block of code 
                                    self.nn.bc+=1 
                                    # add 1 to the bc attribute of the nn object 
                        except Exception as e: 
                        # handle any exception raised in the previous block of code and assign it to e
                            raise e # re-raise e
            if hasattr(self.platform,'DType'): # check if the platform attribute has DType attribute, which indicates tensorflow platform
                loss=total_loss.numpy()/batches # convert total_loss to numpy array, divide it by batches, and assign it to loss
            else: # check if the platform attribute does not have DType attribute, which indicates pytorch platform
                loss=total_loss.detach().numpy()/batches # detach total_loss from computation graph, convert it to numpy array, divide it by batches, and assign it to loss
            if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                train_acc=total_acc.numpy()/batches # convert total_acc to numpy array, divide it by batches, and assign it to train_acc
            self.train_loss=loss # assign loss to self.train_loss
            self.train_loss_list.append(loss) # append loss to self.train_loss_list
            if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                self.train_acc=train_acc # assign train_acc to self.train_acc
                self.train_acc_list.append(train_acc) # append train_acc to self.train_acc_list
            if self.test_flag==True: # check if the test_flag attribute is True
                if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch) # call test method with self.test_data, self.test_labels, and test_batch as arguments and assign the results to self.test_loss and self.test_acc
                    self.test_loss_list.append(self.test_loss) # append self.test_loss to self.test_loss_list
                    self.test_acc_list.append(self.test_acc) # append self.test_acc to self.test_acc_list
                else: # check if the nn object does not have accuracy method
                    self.test_loss=self.test(self.test_data,self.test_labels,test_batch) # call test method with self.test_data, self.test_labels, and test_batch as arguments and assign the result to self.test_loss
                    self.test_loss_list.append(self.test_loss) # append self.test_loss to self.test_loss_list
        else: # check if batch is None
            output,train_loss=self.opt(self.train_data,self.train_labels) # call opt method with self.train_data and self.train_labels as arguments and assign the results to output and train_loss
            self.loss_acc(output=output,labels_batch=labels_batch,loss=train_loss,test_batch=test_batch,total_loss=total_loss,total_acc=total_acc) 
            # call loss_acc method with output, labels_batch, train_loss, test_batch, total_loss, and total_acc as arguments
        return # return from the method
    
    
    def train(self,batch=None,epoch=None,test_batch=None,save=None,one=True,p=None,s=None): 
        # define a method for training on multiple epochs or iterations
        self.batch=batch # assign batch argument to batch attribute
        self.epoch=0 # set epoch attribute to 0
        self.train_counter+=1 # add 1 to train_counter attribute
        if p==None: # check if p argument is None
            self.p=9 # set p attribute to 9
        else: # check if s argument is not None
            self.p=p-1 # subtract 1 from p argument and assign it to p attribute
        if s==None: # check if s argument is None
            self.s=1 # set s attribute to 1
            self.file_list=None # set file_list attribute to None
        else: # check if s argument is not None
            self.s=s-1 # subtract 1 from s argument and assign it to s attribute
            self.file_list=[] # set file_list attribute to an empty list
        if epoch!=None: # check if epoch argument is not None
            for i in range(epoch): # iterate over epoch argument
                t1=time.time() # get the current time and assign it to t1
                self._train(batch,test_batch) # call _train method with batch and test_batch as arguments
                if self.stop_flag==True: # check if stop_flag attribute is True
                    return # return from the method
                if hasattr(self.nn,'ec'): # check if the nn object has ec attribute
                    try: # try the following block of code
                        self.nn.ec.assign_add(1) # call assign_add method of the ec attribute of the nn object with 1 as argument
                    except Exception: # handle any exception raised in the previous block of code
                        self.nn.ec+=1 # add 1 to the ec attribute of the nn object
                self.total_epoch+=1 # add 1 to total_epoch attribute
                if epoch%10!=0: # check if epoch argument is not divisible by 10
                    p=epoch-epoch%self.p # calculate the nearest multiple of p attribute that is less than or equal to epoch argument and assign it to p
                    p=int(p/self.p) # divide p by p attribute and convert it to integer and assign it to p
                    s=epoch-epoch%self.s # calculate the nearest multiple of s attribute that is less than or equal to epoch argument and assign it to s
                    s=int(s/self.s) # divide s by s attribute and convert it to integer and assign it to s
                else: # check if epoch argument is divisible by 10
                    p=epoch/(self.p+1) # divide epoch argument by p attribute plus 1 and assign it to p
                    p=int(p) # convert p to integer and assign it to p
                    s=epoch/(self.s+1) # divide epoch argument by s attribute plus 1 and assign it to s
                    s=int(s) # convert s to integer and assign it to s
                if p==0: # check if p is 0
                    p=1 # set p to 1
                if s==0: # check if s is 0
                    s=1 # set s to 1
                if i%p==0: # check if i is divisible by p 
                    if self.test_flag==False: 
                    # check if test_flag attribute is False 
                        if hasattr(self.nn,'accuracy'): 
                        # check if the nn object has accuracy method 
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss)) 
                            # print the epoch number and train_loss attribute with six decimal places 
                            if self.acc_flag=='%': 
                            # check if acc_flag attribute is '%' 
                                print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100)) 
                                # print the epoch number and train_acc attribute multiplied by 100 with one decimal place 
                            else: 
                            # check if acc_flag attribute is not '%' 
                                print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc)) 
                                # print the epoch number and train_acc attribute with six decimal places 
                            print() 
                            # print a blank line 
                        else: 
                        # check if the nn object does not have accuracy method 
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss)) 
                            # print the epoch number and train_loss attribute with six decimal places 
                            print() 
                            # print a blank line 
                    else: 
                    # check if test_flag attribute is True 
                        if hasattr(self.nn,'accuracy'): 
                        # check if the nn object has accuracy method 
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss)) 
                            # print the epoch number, train_loss attribute, and test_loss attribute with six decimal places 
                            if self.acc_flag=='%': 
                            # check if acc_flag attribute is '%' 
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100)) 
                                # print the epoch number, train_acc attribute multiplied by 100, and test_acc attribute multiplied by 100 with one decimal place 
                            else: 
                            # check if acc_flag attribute is not '%' 
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc)) 
                                # print the epoch number, train_acc attribute, and test_acc attribute with one decimal place 
                            print() 
                            # print a blank line 
                        else: 
                        # check if the nn object does not have accuracy method 
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss)) 
                            # print the epoch number, train_loss attribute, and test_loss attribute with six decimal places 
                            print() 
                            # print a blank line 
                if save!=None and i%s==0: # check if save argument is not None and i is divisible by s
                    self.save(self.total_epoch,one) # call save method with total_epoch attribute and one argument as arguments
                t2=time.time() # get the current time and assign it to t2
                self.time+=(t2-t1) # subtract t1 from t2 and add it to time attribute
        else: # check if epoch argument is None
            i=0 # set i to 0
            while True: # start an infinite loop
                t1=time.time() # get the current time and assign it to t1
                self._train(test_batch=test_batch) # call _train method with test_batch as argument
                if self.stop_flag==True: # check if stop_flag attribute is True
                    return # return from the method
                i+=1 # add 1 to i
                if hasattr(self.nn,'ec'): # check if the nn object has ec attribute
                    try: # try the following block of code
                        self.nn.ec.assign_add(1) # call assign_add method of the ec attribute of the nn object with 1 as argument
                    except Exception: # handle any exception raised in the previous block of code
                        self.nn.ec+=1 # add 1 to the ec attribute of the nn object
                self.total_epoch+=1 # add 1 to total_epoch attribute
                if epoch%10!=0: # check if epoch argument is not divisible by 10
                    p=epoch-epoch%self.p # calculate the nearest multiple of p attribute that is less than or equal to epoch argument and assign it to p
                    p=int(p/self.p) # divide p by p attribute and convert it to integer and assign it to p
                    s=epoch-epoch%self.s # calculate the nearest multiple of s attribute that is less than or equal to epoch argument and assign it to s
                    s=int(s/self.s) # divide s by s attribute and convert it to integer and assign it to s
                else: # check if epoch argument is divisible by 10
                    p=epoch/(self.p+1) # divide epoch argument by p attribute plus 1 and assign it to p
                    p=int(p) # convert p to integer and assign it to p
                    s=epoch/(self.s+1) # divide epoch argument by s attribute plus 1 and assign it to s
                    s=int(s) # convert s to integer and assign it to s
                if p==0: # check if p is 0
                    p=1 # set p to 1
                if s==0: # check if s is 0
                    s=1 # set s to 1
                if i%p==0: # check if i is divisible by p 
                    if self.test_flag==False: 
                    # check if test_flag attribute is False 
                        if hasattr(self.nn,'accuracy'): 
                        # check if the nn object has accuracy method 
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss)) 
                            # print the epoch number and train_loss attribute with six decimal places 
                            if self.acc_flag=='%': 
                            # check if acc_flag attribute is '%' 
                                print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100)) 
                                # print the epoch number and train_acc attribute multiplied by 100 with one decimal place 
                            else: 
                            # check if acc_flag attribute is not '%' 
                                print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc)) 
                                # print the epoch number and train_acc attribute with six decimal places 
                            print() 
                            # print a blank line 
                        else: 
                        # check if the nn object does not have accuracy method 
                            print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss)) 
                            # print the epoch number and train_loss attribute with six decimal places 
                            print() 
                            # print a blank line 
                    else: 
                    # check if test_flag attribute is True 
                        if hasattr(self.nn,'accuracy'): 
                        # check if the nn object has accuracy method 
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss)) 
                            # print the epoch number, train_loss attribute, and test_loss attribute with six decimal places 
                            if self.acc_flag=='%': 
                            # check if acc_flag attribute is '%' 
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100)) 
                                # print the epoch number, train_acc attribute multiplied by 100, and test_acc attribute multiplied by 100 with one decimal place 
                            else: 
                            # check if acc_flag attribute is not '%' 
                                print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc)) 
                                # print the epoch number, train_acc attribute, and test_acc attribute with one decimal place 
                            print() 
                            # print a blank line 
                        else: 
                        # check if the nn object does not have accuracy method 
                            print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss)) 
                            # print the epoch number, train_loss attribute, and test_loss attribute with six decimal places 
                            print() 
                            # print a blank line 
                if save!=None and i%s==0: # check if save argument is not None and i is divisible by s
                    self.save(self.total_epoch,one) # call save method with total_epoch attribute and one argument as arguments
                t2=time.time() # get the current time and assign it to t2
                self.time+=(t2-t1) # subtract t1 from t2 and add it to time attribute
        if save!=None: # check if save argument is not None
            self.save() # call save method
        self._time=self.time-int(self.time) # subtract the integer part of time attribute from time attribute and assign it to _time attribute
        if self._time<0.5: # check if _time attribute is less than 0.5
            self.time=int(self.time) # convert time attribute to integer and assign it to time attribute
        else: # check if _time attribute is not less than 0.5
            self.time=int(self.time)+1 # convert time attribute to integer, add 1, and assign it to time attribute
        self.total_time+=self.time # add time attribute to total_time attribute
        if self.test_flag==False: # check if test_flag attribute is False
            print('last loss:{0:.6f}'.format(self.train_loss)) # print the last train_loss attribute with six decimal places
        else: # check if test_flag attribute is True
            print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss)) # print the last train_loss attribute and last test_loss attribute with six decimal places
        if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
            if self.acc_flag=='%': # check if acc_flag attribute is '%'
                if self.test_flag==False: # check if test_flag attribute is False
                    print('last accuracy:{0:.1f}'.format(self.train_acc*100)) # print the last train_acc attribute multiplied by 100 with one decimal place
                else: # check if test_flag attribute is True
                    print('last accuracy:{0:.1f},last test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100)) # print the last train_acc attribute multiplied by 100 and last test_acc attribute multiplied by 100 with one decimal place
            else: # check if acc_flag attribute is not '%'
                if self.test_flag==False: # check if test_flag attribute is False
                    print('last accuracy:{0:.6f}'.format(self.train_acc)) # print the last train_acc attribute with six decimal places
                else: # check if test_flag attribute is True
                    print('last accuracy:{0:.6f},last test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc)) # print the last train_acc attribute and last test_acc attribute with six decimal places   
        print() # print a blank line
        print('time:{0}s'.format(self.time)) # print the time attribute with 's' as unit
        self.training_flag=False # set training_flag attribute to False
        return # return from the method
    
    
    def train_online(self): 
        # define a method for online training
        while True: # start an infinite loop
            if hasattr(self.nn,'save'): # check if the nn object has save method
                self.nn.save(self.save) # call save method of the nn object with save attribute as argument
            if hasattr(self.nn,'stop_flag'): # check if the nn object has stop_flag attribute
                if self.nn.stop_flag==True: # check if stop_flag attribute of the nn object is True
                    return # return from the method
            if hasattr(self.nn,'stop_func'): # check if the nn object has stop_func method
                if self.nn.stop_func(): # call stop_func method of the nn object and check if it returns True
                    return # return from the method
            if hasattr(self.nn,'suspend_func'): # check if the nn object has suspend_func method
                self.nn.suspend_func() # call suspend_func method of the nn object
            data=self.nn.online() # call online method of the nn object and assign it to data
            if data=='stop': # check if data is 'stop'
                return # return from the method
            elif data=='suspend': # check if data is 'suspend'
                self.nn.suspend_func() # call suspend_func method of the nn object
            output,loss=self.opt(data[0],data[1]) # call opt method with data[0] and data[1] as arguments and assign the results to output and loss
            loss=loss.numpy() # convert loss to numpy array and assign it to loss
            if len(self.nn.train_loss_list)==self.nn.max_length: # check if the length of train_loss_list attribute of the nn object is equal to max_length attribute of the nn object
                del self.nn.train_loss_list[0] # delete the first element of train_loss_list attribute of the nn object
            self.nn.train_loss_list.append(loss) # append loss to train_loss_list attribute of the nn object
            try: # try the following block of code
                if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                    train_acc=self.nn.accuracy(output,data[1]) # call accuracy method of the nn object with output and data[1] as arguments and assign it to train_acc
                    if len(self.nn.train_acc_list)==self.nn.max_length: # check if the length of train_acc_list attribute of the nn object is equal to max_length attribute of the nn object
                        del self.nn.train_acc_list[0] # delete the first element of train_acc_list attribute of the nn object
                    self.train_acc_list.append(train_acc) # append train_acc to train_acc_list attribute of the nn object
            except Exception as e: # handle any exception raised in the previous block of code and assign it to e
                raise e # re-raise e
            if hasattr(self.nn,'counter'): # check if the nn object has counter attribute
                self.nn.counter+=1 # add 1 to counter attribute of the nn object
        return # return from the method
    
    
    @function(jit_compile=True) 
    # use function decorator with jit_compile argument set to True for just-in-time compilation of this method for faster execution 
    def test_tf(self,data,labels): 
        # define a method for testing using tensorflow platform 
        try: 
        # try the following block of code 
            try: 
            # try the following block of code 
                output=self.nn.fp(data) 
                # call fp method of the nn object with data as argument and assign it to output 
                loss=self.nn.loss(output,labels) 
                # call loss method of the nn object with output and labels as arguments and assign it to loss 
            except Exception: 
            # handle any exception raised in the previous block of code 
                output,loss=self.nn.fp(data,labels) 
                # call fp method of the nn object with data and labels as arguments and assign the results to output and loss 
        except Exception as e: 
        # handle any exception raised in the previous block of code and assign it to e 
            raise e # re-raise e
        try: # try the following block of code
            acc=self.nn.accuracy(output,labels) # call accuracy method of the nn object with output and labels as arguments and assign it to acc
        except Exception as e: # handle any exception raised in the previous block of code and assign it to e
            if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                raise e # re-raise e
            else: # check if the nn object does not have accuracy method
                acc=None # set acc to None
        return loss,acc # return loss and acc
    
    
    def test_pytorch(self,data,labels): 
        # define a method for testing using pytorch platform 
        output=self.nn.fp(data) # call fp method of the nn object with data as argument and assign it to output
        loss=self.nn.loss(output,labels) # call loss method of the nn object with output and labels as arguments and assign it to loss
        try: # try the following block of code
            acc=self.nn.accuracy(output,labels) # call accuracy method of the nn object with output and labels as arguments and assign it to acc
        except Exception as e: # handle any exception raised in the previous block of code and assign it to e
            if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                raise e # re-raise e
            else: # check if the nn object does not have accuracy method
                acc=None # set acc to None
        return loss,acc # return loss and acc
    
    
    def test(self,test_data=None,test_labels=None,batch=None): 
        # define a method for testing on a given data set 
        if test_data is not None and type(self.nn.param[0])!=list: 
        # check if test_data is not None and the type of the first element of self.nn.param is not list 
            test_data=test_data.astype(self.nn.param[0].dtype.name) 
            # convert the test_data to the same data type as the first element of self.nn.param and assign it to test_data 
            test_labels=test_labels.astype(self.nn.param[0].dtype.name) 
            # convert the test_labels to the same data type as the first element of self.nn.param and assign it to test_labels 
        elif test_data is not None: 
        # check if test_data is not None 
            test_data=test_data.astype(self.nn.param[0][0].dtype.name) 
            # convert the test_data to the same data type as the first element of the first element of self.nn.param and assign it to test_data 
            test_labels=test_labels.astype(self.nn.param[0][0].dtype.name) 
            # convert the test_labels to the same data type as the first element of the first element of self.nn.param and assign it to test_labels 
        if self.process_t!=None: 
        # check if process_t attribute is not None 
            if self.prefetch_batch_size_t==None: 
            # check if prefetch_batch_size_t attribute is None 
                parallel_test_=parallel_test(self.nn,self.test_data,self.test_labels,self.process_t,batch,test_dataset=self.test_dataset) 
                # create a parallel_test object with self.nn, self.test_data, self.test_labels, self.process_t, batch, and self.test_dataset as arguments and assign it to parallel_test_ 
            else: 
            # check if prefetch_batch_size_t attribute is not None 
                parallel_test_=parallel_test(self.nn,self.test_data,self.test_labels,self.process_t,batch,self.prefetch_batch_size_t,self.test_dataset) 
                # create a parallel_test object with self.nn, self.test_data, self.test_labels, self.process_t, batch, self.prefetch_batch_size_t, and self.test_dataset as arguments and assign it to parallel_test_ 
            if type(self.test_data)!=list: 
            # check if the type of test_data attribute is not list 
                parallel_test_.segment_data() 
                # call segment_data method of parallel_test_ object 
            for p in range(self.process_t): 
            # iterate over process_t attribute 
                Process(target=parallel_test_.test).start() 
                # create a Process object with target argument set to test method of parallel_test_ object and call start method on it 
            try: 
            # try the following block of code 
                if hasattr(self.nn,'accuracy'): 
                # check if the nn object has accuracy method 
                    test_loss,test_acc=parallel_test_.loss_acc() 
                    # call loss_acc method of parallel_test_ object and assign the results to test_loss and test_acc 
            except Exception as e: 
            # handle any exception raised in the previous block of code and assign it to e 
                if hasattr(self.nn,'accuracy'): 
                # check if the nn object has accuracy method 
                    raise e 
                    # re-raise e 
                else: 
                # check if the nn object does not have accuracy method 
                    test_loss=parallel_test_.loss_acc() 
                    # call loss_acc method of parallel_test_ object and assign the result to test_loss 
        elif batch!=None: 
        # check if batch argument is not None 
            total_loss=0 
            # initialize total_loss to 0 
            total_acc=0 
            # initialize total_acc to 0 
            if self.test_dataset!=None: 
            # check if test_dataset attribute is not None 
                batches=0 
                # initialize batches to 0 
                for data_batch,labels_batch in self.test_dataset: 
                # iterate over test_dataset attribute 
                    batches+=1 
                    # add 1 to batches 
                    if hasattr(self.platform,'DType'): 
                    # check if the platform attribute has DType attribute, which indicates tensorflow platform
                        batch_loss,batch_acc=self.test_tf(data_batch,labels_batch) 
                        # call test_tf method with data_batch and labels_batch as arguments and assign the results to batch_loss and batch_acc
                    else: 
                    # check if the platform attribute does not have DType attribute, which indicates pytorch platform
                        batch_loss,batch_acc=self.test_pytorch(data_batch,labels_batch) 
                        # call test_pytorch method with data_batch and labels_batch as arguments and assign the results to batch_loss and batch_acc
                    total_loss+=batch_loss # add batch_loss to total_loss
                    try: # try the following block of code
                        total_acc+=batch_acc # add batch_acc to total_acc
                    except Exception: # handle any exception raised in the previous block of code
                        pass # do nothing
                test_loss=total_loss.numpy()/batches # convert total_loss to numpy array, divide it by batches, and assign it to test_loss
                if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                    test_acc=total_acc.numpy()/batches # convert total_acc to numpy array, divide it by batches, and assign it to test_acc
            else: # check if test_dataset attribute is None
                shape0=test_data.shape[0] # get the first dimension of test_data and assign it to shape0
                batches=int((shape0-shape0%batch)/batch) # calculate the number of batches and assign it to batches
                for j in range(batches): # iterate over batches
                    index1=j*batch # calculate the start index of a batch and assign it to index1
                    index2=(j+1)*batch # calculate the end index of a batch and assign it to index2
                    data_batch=test_data[index1:index2] # slice the test_data from index1 to index2 and assign it to data_batch
                    labels_batch=test_labels[index1:index2] # slice the test_labels from index1 to index2 and assign it to labels_batch
                    if hasattr(self.platform,'DType'): # check if the platform attribute has DType attribute, which indicates tensorflow platform
                        batch_loss,batch_acc=self.test_tf(data_batch,labels_batch) # call test_tf method with data_batch and labels_batch as arguments and assign the results to batch_loss and batch_acc
                    else: # check if the platform attribute does not have DType attribute, which indicates pytorch platform
                        batch_loss,batch_acc=self.test_pytorch(data_batch,labels_batch) # call test_pytorch method with data_batch and labels_batch as arguments and assign the results to batch_loss and batch_acc
                    total_loss+=batch_loss # add batch_loss to total_loss
                    if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                        total_acc+=batch_acc # add batch_acc to total_acc
                if shape0%batch!=0: # check if there is a remainder after dividing shape0 by batch
                    batches+=1 # add 1 to batches
                    index1=batches*batch # calculate the start index of a batch and assign it to index1
                    index2=batch-(shape0-batches*batch) # calculate the end index of a batch and assign it to index2
                    try: # try the following block of code
                        try: # try the following block of code
                            data_batch=self.platform.concat([test_data[index1:],test_data[:index2]],0) 
                            # concatenate the test_data from index1 to the end and from the beginning to index2 along the first axis and assign it to data_batch 
                            labels_batch=self.platform.concat([test_labels[index1:],test_labels[:index2]],0) 
                            # concatenate the test_labels from index1 to the end and from the beginning to index2 along the first axis and assign it to labels_batch 
                        except Exception: # handle any exception raised in the previous block of code
                            data_batch=np.concatenate([test_data[index1:],test_data[:index2]],0) 
                            # concatenate the test_data from index1 to the end and from the beginning to index2 along the first axis using numpy and assign it to data_batch 
                            labels_batch=np.concatenate([test_labels[index1:],test_labels[:index2]],0) 
                            # concatenate the test_labels from index1 to the end and from the beginning to index2 along the first axis using numpy and assign it to labels_batch 
                    except Exception as e: # handle any exception raised in the previous block of code and assign it to e
                        raise e # re-raise e
                    if hasattr(self.platform,'DType'): # check if the platform attribute has DType attribute, which indicates tensorflow platform
                        batch_loss,batch_acc=self.test_tf(data_batch,labels_batch) # call test_tf method with data_batch and labels_batch as arguments and assign the results to batch_loss and batch_acc
                    else: # check if the platform attribute does not have DType attribute, which indicates pytorch platform
                        batch_loss,batch_acc=self.test_pytorch(data_batch,labels_batch) # call test_pytorch method with data_batch and labels_batch as arguments and assign the results to batch_loss and batch_acc
                    total_loss+=batch_loss # add batch_loss to total_loss
                    if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                        total_acc+=batch_acc # add batch_acc to total_acc
                if hasattr(self.platform,'DType'): # check if the platform attribute has DType attribute, which indicates tensorflow platform
                    test_loss=total_loss.numpy()/batches # convert total_loss to numpy array, divide it by batches, and assign it to test_loss
                else: # check if the platform attribute does not have DType attribute, which indicates pytorch platform
                    test_loss=total_loss.detach().numpy()/batches # detach total_loss from computation graph, convert it to numpy array, divide it by batches, and assign it to test_loss
                if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                    test_acc=total_acc.numpy()/batches # convert total_acc to numpy array, divide it by batches, and assign it to test_acc
        else: # check if batch argument is None
            if hasattr(self.platform,'DType'): # check if the platform attribute has DType attribute, which indicates tensorflow platform
                batch_loss,batch_acc=self.test_tf(test_data,test_labels) # call test_tf method with test_data and test_labels as arguments and assign the results to batch_loss and batch_acc
            else: # check if the platform attribute does not have DType attribute, which indicates pytorch platform
                batch_loss,batch_acc=self.test_pytorch(test_data,test_labels) # call test_pytorch method with test_data and test_labels as arguments and assign the results to batch_loss and batch_acc
            test_loss=batch_loss.numpy() # convert batch_loss to numpy array and assign it to test_loss
            if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
                test_acc=batch_acc.numpy() # convert batch_acc to numpy array and assign it to test_acc
        if hasattr(self.nn,'accuracy'): # check if the nn object has accuracy method
            return test_loss,test_acc # return test_loss and test_acc
        else: # check if the nn object does not have accuracy method
            return test_loss # return test_loss
