import tensorflow as tf # import TensorFlow library
import numpy as np # import NumPy library
from multiprocessing import Array # import Array class from multiprocessing module
import numpy.ctypeslib as npc # import ctypeslib module from NumPy


@tf.function(jit_compile=True) # use TensorFlow's just-in-time compilation decorator
def test_tf(nn,data,labels): # define a function that tests a TensorFlow network
    try: # try to execute the following block of code
        try: # try to execute the following block of code
            output=nn.fp(data) # get the output of the network by calling its forward propagation method
            loss=nn.loss(output,labels) # calculate the loss by calling its loss function
        except Exception: # if an exception occurs in the previous block
            output,loss=nn.fp(data,labels) # get the output and loss of the network by calling its forward propagation method with both data and labels as arguments
    except Exception as e: # if an exception occurs in the previous block
        raise e # re-raise the exception
    try: # try to execute the following block of code
        if hasattr(nn,'accuracy'): # check if the network has an accuracy attribute
            acc=nn.accuracy(output,labels) # calculate the accuracy by calling its accuracy function
        else: # if the network does not have an accuracy attribute
            acc=None # set the accuracy to None
    except Exception as e: # if an exception occurs in the previous block
        raise e # re-raise the exception
    return loss,acc # return the loss and accuracy


def test_pytorch(nn,data,labels): # define a function that tests a PyTorch network
    try: # try to execute the following block of code
        try: # try to execute the following block of code
            output=nn.fp(data) # get the output of the network by calling its forward propagation method
            loss=nn.loss(output,labels) # calculate the loss by calling its loss function
        except Exception: # if an exception occurs in the previous block
            output,loss=nn.fp(data,labels) # get the output and loss of the network by calling its forward propagation method with both data and labels as arguments
    except Exception as e: # if an exception occurs in the previous block
        raise e # re-raise the exception
    try: # try to execute the following block of code
        if hasattr(nn,'accuracy'): # check if the network has an accuracy attribute
            acc=nn.accuracy(output,labels) # calculate the accuracy by calling its accuracy function
        else:  # if the network does not have an accuracy attribute 
          acc=None  # set the accuracy to None 
    except Exception as e:  # if an exception occurs in the previous block 
        raise e  # re-raise the exception 
    return loss,acc  # return the loss and accuracy


def test(nn,test_data,test_labels,platform,batch=None,loss=None,acc_flag='%'):  # define a function that tests a generic network on a given platform 
    if type(nn.param[0])!=list:  # check if the first element of nn.param is not a list 
        test_data=test_data.astype(nn.param[0].dtype.name)  # convert test_data to match the data type of nn.param[0]
        test_labels=test_labels.astype(nn.param[0].dtype.name)  # convert test_labels to match the data type of nn.param[0]
    else:  # if the first element of nn.param is a list 
        test_data=test_data.astype(nn.param[0][0].dtype.name)  # convert test_data to match the data type of nn.param[0][0]
        test_labels=test_labels.astype(nn.param[0][0].dtype.name)  # convert test_labels to match the data type of nn.param[0][0]
    if batch!=None:  # check if batch is not None 
        total_loss=0  # initialize total_loss to zero 
        total_acc=0  # initialize total_acc to zero 
        batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)  # calculate how many batches are needed 
        shape0=test_data.shape[0]  # store the original shape of test_data along axis 0 
        for j in range(batches):  # loop through each batch 
            index1=j*batch  # calculate the starting index of data_batch 
            index2=(j+1)*batch  # calculate the ending index of data_batch 
            data_batch=test_data[index1:index2]  # slice test_data to get data_batch 
            labels_batch=test_labels[index1:index2]  # slice test_labels to get labels_batch 
            if hasattr(platform,'DType'):  # check if platform has a DType attribute (indicating TensorFlow) 
                batch_loss,batch_acc=test_tf(data_batch,labels_batch)  # call test_tf function to get batch_loss and batch_acc 
            else:  # if platform does not have a DType attribute (indicating PyTorch) 
                batch_loss,batch_acc=test_pytorch(data_batch,labels_batch)  # call test_pytorch function to get batch_loss and batch_acc 
            total_loss+=batch_loss  # add batch_loss to total_loss 
            if hasattr(nn,'accuracy'):  # check if the network has an accuracy attribute 
                total_acc+=batch_acc  # add batch_acc to total_acc 
        if shape0%batch!=0:  # check if there is a remainder after dividing test_data by batch 
            batches+=1  # increment batches by one 
            index1=batches*batch  # calculate the starting index of data_batch 
            index2=batch-(shape0-batches*batch)  # calculate the ending index of data_batch 
            try:  # try to execute the following block of code 
                try:  # try to execute the following block of code 
                    data_batch=platform.concat([test_data[index1:],test_data[:index2]],0)  # concatenate the remaining test_data and some test_data from the beginning along axis 0 to get data_batch 
                    labels_batch=platform.concat([test_labels[index1:],test_labels[:index2]],0)  # concatenate the corresponding labels along axis 0 to get labels_batch 
                except Exception:  # if an exception occurs in the previous block 
                    data_batch=np.concatenate([test_data[index1:],test_data[:index2]],0)  # use NumPy's concatenate function instead of platform's concat function
                    labels_batch=np.concatenate([test_labels[index1:],test_labels[:index2]],0)  # use NumPy's concatenate function instead of platform's concat function
            except Exception as e:  # if an exception occurs in the previous block 
                raise e  # re-raise the exception
            if hasattr(platform,'DType'):  # check if platform has a DType attribute (indicating TensorFlow) 
                batch_loss,batch_acc=test_tf(data_batch,labels_batch)  # call test_tf function to get batch_loss and batch_acc
            else:  # if platform does not have a DType attribute (indicating PyTorch) 
                batch_loss,batch_acc=test_pytorch(data_batch,labels_batch)  # call test_pytorch function to get batch_loss and batch_acc
            total_loss+=batch_loss  # add batch_loss to total_loss
            if hasattr(nn,'accuracy'):  # check if the network has an accuracy attribute
                total_acc+=batch_acc  # add batch_acc to total_acc
        test_loss=total_loss.numpy()/batches  # calculate the average test loss by dividing total_loss by batches and converting it to a NumPy array
        test_loss=test_loss.astype(np.float32)  # convert test_loss to float32 type
        if hasattr(nn,'accuracy'):  # check if the network has an accuracy attribute
            test_acc=total_acc.numpy()/batches  # calculate the average test accuracy by dividing total_acc by batches and converting it to a NumPy array
            test_acc=test_acc.astype(np.float32)  # convert test_acc to float32 type
    else:  # if batch is None
        if hasattr(platform,'DType'):  # check if platform has a DType attribute (indicating TensorFlow) 
            batch_loss,batch_acc=test_tf(test_data,test_labels)  # call test_tf function to get batch_loss and batch_acc
        else:  # if platform does not have a DType attribute (indicating PyTorch) 
            batch_loss,batch_acc=test_pytorch(test_data,test_labels)  # call test_pytorch function to get batch_loss and batch_acc
        test_loss=batch_loss.numpy().astype(np.float32)  # convert batch_loss to a NumPy array and float32 type
        if hasattr(nn,'accuracy'):  # check if the network has an accuracy attribute
            test_acc=batch_acc.numpy().astype(np.float32)   # convert batch_acc to a NumPy array and float32 type
    print('test loss:{0:.6f}'.format(test_loss))   # print the test loss with six decimal places
    if hasattr(nn,'accuracy'):   # check if the network has an accuracy attribute
        if acc_flag=='%':   # check if acc_flag is '%' (indicating percentage format)
            print('test acc:{0:.1f}'.format(test_acc*100))   # print the test accuracy with one decimal place and multiply by 100
        else:   # if acc_flag is not '%' (indicating decimal format)
            print('test acc:{0:.6f}'.format(test_acc)) # print the test accuracy with six decimal places
        if acc_flag=='%': # check if acc_flag is '%' (indicating percentage format)
            return test_loss,test_acc*100 # return the test loss and accuracy multiplied by 100
        else: # if acc_flag is not '%' (indicating decimal format)
            return test_loss,test_acc # return the test loss and accuracy
    else: # if the network does not have an accuracy attribute
        return test_loss # return the test loss


class parallel_test: # define a class for parallel testing
    def __init__(self,nn,test_data,test_labels,process,batch,prefetch_batch_size=tf.data.AUTOTUNE,test_dataset=None,): # define the constructor method
        self.nn=nn # store the neural network as an attribute
        if test_data is not None and type(self.nn.param[0])!=list: # check if test_data is not None and the first element of nn.param is not a list
            self.test_data=test_data.astype(self.nn.param[0].dtype.name) # convert test_data to match the data type of nn.param[0]
            self.test_labels=test_labels.astype(self.nn.param[0].dtype.name) # convert test_labels to match the data type of nn.param[0]
        elif test_data is not None: # if test_data is not None but the first element of nn.param is a list
            self.test_data=test_data.astype(self.nn.param[0][0].dtype.name) # convert test_data to match the data type of nn.param[0][0]
            self.test_labels=test_labels.astype(self.nn.param[0][0].dtype.name) # convert test_labels to match the data type of nn.param[0][0]
        self.test_dataset=test_dataset # store the test_dataset as an attribute
        self.process=process # store the number of processes as an attribute
        self.batch=batch # store the batch size as an attribute
        if type(self.nn.param[0])!=list: # check if the first element of nn.param is not a list
            self.loss=Array('f',np.zeros([process],dtype=self.nn.param[0].dtype.name)) # create a shared memory array for storing the loss values for each process with the same data type as nn.param[0]
        else: # if the first element of nn.param is a list
            self.loss=Array('f',np.zeros([process],dtype=self.nn.param[0][0].dtype.name)) # create a shared memory array for storing the loss values for each process with the same data type as nn.param[0][0]
        if hasattr(nn,'accuracy'): # check if the network has an accuracy attribute
            if type(self.nn.param[0])!=list: # check if the first element of nn.param is not a list
                self.acc=Array('f',np.zeros([process],dtype=self.nn.param[0].dtype.name)) # create a shared memory array for storing the accuracy values for each process with the same data type as nn.param[0]
            else: # if the first element of nn.param is a list
                self.acc=Array('f',np.zeros([process],dtype=self.nn.param[0][0].dtype.name)) # create a shared memory array for storing the accuracy values for each process with the same data type as nn.param[0][0]
        self.prefetch_batch_size=prefetch_batch_size # store the prefetch batch size as an attribute
    
    
    def segment_data(self): # define a method that segments the test data and labels into equal parts for each process
        if len(self.test_data)!=self.process: # check if the length of test_data is not equal to the number of processes
            length=len(self.test_data)-len(self.test_data)%self.process # calculate the length of test_data that can be evenly divided by the number of processes
            data=self.test_data[:length] # slice test_data to get only that part
            labels=self.test_labels[:length] # slice test_labels to get only that part
            data=np.split(data,self.process) # split data into equal parts for each process
            labels=np.split(labels,self.process) # split labels into equal parts for each process
            self.test_data=data # assign data to test_data attribute
            self.test_labels=labels # assign labels to test_labels attribute
        return
    
    
    @tf.function(jit_compile=True) # use TensorFlow's just-in-time compilation decorator
    def test_(self,data,labels,p): # define a method that tests a TensorFlow network on a given data, labels, and process index
        try: # try to execute the following block of code
            try: # try to execute the following block of code
                try: # try to execute the following block of code
                    output=self.nn.fp(data,p) # get the output of the network by calling its forward propagation method with data and process index as arguments
                    loss=self.nn.loss(output,labels,p) # calculate the loss by calling its loss function with output, labels, and process index as arguments
                except Exception:  # if an exception occurs in the previous block 
                    output,loss=self.nn.fp(data,labels,p)  # get the output and loss of the network by calling its forward propagation method with data, labels, and process index as arguments
            except Exception:  # if an exception occurs in the previous block 
                try:  # try to execute the following block of code 
                    output=self.nn.fp(data)  # get the output of the network by calling its forward propagation method with data as argument
                    loss=self.nn.loss(output,labels)  # calculate the loss by calling its loss function with output and labels as arguments
                except Exception:  # if an exception occurs in the previous block 
                    output,loss=self.nn.fp(data,labels)  # get the output and loss of the network by calling its forward propagation method with data and labels as arguments
        except Exception as e:  # if an exception occurs in the previous block 
            raise e  # re-raise the exception
        try:  # try to execute the following block of code 
            if hasattr(self.nn,'accuracy'):  # check if the network has an accuracy attribute 
                try:  # try to execute the following block of code 
                    acc=self.nn.accuracy(output,labels,p)  # calculate the accuracy by calling its accuracy function with output, labels, and process index as arguments
                except Exception:  # if an exception occurs in the previous block 
                    acc=self.nn.accuracy(output,labels)  # calculate the accuracy by calling its accuracy function with output and labels as arguments
            else:  # if the network does not have an accuracy attribute 
                acc=None  # set the accuracy to None
        except Exception as e:  # if an exception occurs in the previous block 
            raise e  # re-raise the exception
        return loss,acc  # return the loss and accuracy
    
    
    def test(self,p): # define a method that tests a generic network on a given process index
        if self.test_dataset is None: # check if test_dataset is None
            test_ds=tf.data.Dataset.from_tensor_slices((self.test_data[p],self.test_labels[p])).batch(self.batch).prefetch(self.prefetch_batch_size) # create a TensorFlow dataset from test_data and test_labels for the given process index, batch it, and prefetch it
        elif self.test_dataset is not None and type(self.test_dataset)==list: # check if test_dataset is not None and is a list
            test_ds=self.test_dataset[p] # get the test_dataset for the given process index
        else: # if test_dataset is not None and is not a list
            test_ds=self.test_dataset # use test_dataset as it is
        for data_batch,labels_batch in test_ds: # loop through each batch of data and labels in test_ds
            try: # try to execute the following block of code
                batch_loss,batch_acc=self.test_(data_batch,labels_batch,p) # call test_ method to get batch_loss and batch_acc
            except Exception as e: # if an exception occurs in the previous block
                raise e # re-raise the exception
            if hasattr(self.nn,'accuracy'): # check if the network has an accuracy attribute
                self.loss[p]+=batch_loss # add batch_loss to loss array for the given process index
                self.acc[p]+=batch_acc # add batch_acc to acc array for the given process index
            else: # if the network does not have an accuracy attribute
                self.loss[p]+=batch_loss # add batch_loss to loss array for the given process index
        return
    
    
    def loss_acc(self): # define a method that calculates and returns the average loss and accuracy across all processes
        if self.test_dataset is None: # check if test_dataset is None
            shape=len(self.test_data[0])*self.process # calculate the total number of data points by multiplying the length of test_data for one process by the number of processes
        elif self.test_dataset is not None and type(self.test_dataset)==list: # check if test_dataset is not None and is a list
            shape=len(self.test_dataset[0])*len(self.test_dataset) # calculate the total number of data points by multiplying the length of test_dataset for one process by the length of test_dataset list
        else: # if test_dataset is not None and is not a list
            shape=len(self.test_dataset)*self.process # calculate the total number of data points by multiplying the length of test_dataset by the number of processes
        batches=int((shape-shape%self.batch)/self.batch) # calculate how many batches are needed by dividing shape by batch size and rounding down
        if shape%self.batch!=0: # check if there is a remainder after dividing shape by batch size
            batches+=1 # increment batches by one
        if hasattr(self.nn,'accuracy'): # check if the network has an accuracy attribute
            return np.sum(npc.as_array(self.loss.get_obj()))/batches,np.sum(npc.as_array(self.acc.get_obj()))/batches # return the average loss and accuracy by summing up the loss and acc arrays and dividing them by batches
        else: # if the network does not have an accuracy attribute
            return np.sum(npc.as_array(self.loss.get_obj()))/batches # return the average loss by summing up the loss array and dividing it by batches