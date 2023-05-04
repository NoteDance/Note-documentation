from tensorflow import function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time


#You can analyze kernel by example.
'''
example:
import kernel_reduced as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train)   #input you data
kernel.train(32,5)         #train neural network
                           #batch size:32
                           #epoch:5


multithreading example:
import kernel_reduced as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=7                        #thread count,use 7 threads to train
kernel.epoch_=6                #epoch:6
kernel.PO=2                    #use PO2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32) #batch size:32
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
    

multiprocessing example:
import kernel_reduced as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
from multiprocessing import Process,Lock
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=7                        #thread count,use 7 processes to train
kernel.epoch_=6                #epoch:6
kernel.PO=2                    #use PO2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[Lock(),Lock(),Lock()]
for _ in range(7):
	p=Process(target=kernel.train(32)) #batch size:32
	p.start()
for _ in range(7):
	p.join()
'''
class kernel:
    def __init__(self,nn=None):
        self.nn=nn  #Neural network object.
        self.platform=None  #Platform object,kernel use it to distinguish platform you use.
        self.PO=None  #PO object,three parallel optimization methods correspond to three numbers.
        self.lock=None  #External incoming lock.
        self.process_thread=None  #Process thread object,threads or processes count you use.
        self.multiprocessing_threading=None
        self.process_thread_counter=0
        self.train_ds=None
        self.data_segment_flag=False
        self.batches=None
        self.buffer_size=None
        self.epoch_=None  #Training epoch count.
        self.epoch_counter=0
        self.save_flag=False
        self.save_epoch=None
        self.batch=None
        self.epoch=0
        self.acc_flag='%'
        self.train_counter=0
        self.opt_counter=None
        self.filename='save.dat'
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.test_flag=False
        self.total_epoch=0
        self.time=0
        self.total_time=0
    
    
    #Turn training data into kernel's instance object.
    def data(self,train_data=None,train_labels=None,test_data=None,test_labels=None,train_dataset=None,test_dataset=None):
        self.train_data=train_data
        self.train_labels=train_labels
        self.train_dataset=train_dataset
        if self.data_segment_flag==True:
            self.train_data,self.train_labels=self.segment_data()
        self.test_data=test_data
        self.test_labels=test_labels
        self.test_dataset=test_dataset
        try:
            if test_data==None:
                self.test_flag=False
        except ValueError:
            self.test_flag=True
        if self.train_dataset==None:
            if type(self.train_data)==list:
                self.shape0=train_data[0].shape[0]
            else:
                self.shape0=train_data.shape[0]
        if self.train_counter==0 and self.process_thread!=None:
            if type(self.process_thread)==list:
                self.process_thread_num=np.arange(self.process_thread[0])
                self.process_thread_num=list(self.process_thread_num)
                self.thread_num=np.arange(self.process_thread[0]*self.process_thread[1]).reshape(self.process_thread[0],self.process_thread[1])
                self.thread_num=list(self.thread_num)
                self.thread_num=[list(self.thread_num[i]) for i in range(len(self.thread_num))]
            else:
                self.process_thread_num=np.arange(self.process_thread)
                self.process_thread_num=list(self.process_thread_num)
            if self.epoch_!=None:
                if type(self.process_thread)==list:
                    self.batch_counter=np.zeros([self.process_thread[0]*self.process_thread[1]],dtype=np.int32)
                    self.total_loss=np.zeros([self.process_thread[0]*self.process_thread[1]],dtype=np.float32)
                else:
                    self.batch_counter=np.zeros(self.process_thread,dtype=np.int32)
                    self.total_loss=np.zeros(self.process_thread,dtype=np.float32)
                try:
                    if self.nn.accuracy!=None:
                        if type(self.process_thread)==list:
                            self.total_acc=np.zeros([self.process_thread[0]*self.process_thread[1]],dtype=np.float32)
                        else:
                            self.total_acc=np.zeros(self.process_thread,dtype=np.float32)
                except AttributeError:
                    pass
            try:
                if self.nn.attenuate!=None:
                    if type(self.process_thread)==list:
                        self.opt_counter=np.zeros([self.process_thread[0]*self.process_thread[1]],dtype=np.float32)
                    else:
                        self.opt_counter=np.zeros(self.process_thread,dtype=np.float32)
            except AttributeError:
                pass
            try:
                if type(self.process_thread)==list:
                    self.nn.bc=np.zeros([self.process_thread[0]*self.process_thread[1]],dtype=np.float32)
                else:
                    self.nn.bc=np.zeros(self.process_thread,dtype=np.float32)
            except AttributeError:
                pass
        return
    
    
    def segment_data(self):
        if len(self.train_data)!=self.process_thread:
            data=None
            labels=None
            segments=int((len(self.train_data)-len(self.train_data)%self.process_thread)/self.process_thread)
            for i in range(self.process_thread):
                index1=i*segments
                index2=(i+1)*segments
                if i==0:
                    data=np.expand_dims(self.train_data[index1:index2],axis=0)
                    labels=np.expand_dims(self.train_labels[index1:index2],axis=0)
                else:
                    data=np.concatenate((data,np.expand_dims(self.train_data[index1:index2],axis=0)))
                    labels=np.concatenate((labels,np.expand_dims(self.train_labels[index1:index2],axis=0)))
            if len(data)%self.process_thread!=0:
                segments+=1
                index1=segments*self.process_thread
                index2=self.process_thread-(len(self.train_data)-segments*self.process_thread)
                data=np.concatenate((data,np.expand_dims(self.train_data[index1:index2],axis=0)))
                labels=np.concatenate((labels,np.expand_dims(self.train_labels[index1:index2],axis=0)))
            return data,labels
    
    
    #loss_acc function be used for calculating total loss and total acc.
    def loss_acc(self,output=None,labels_batch=None,loss=None,test_batch=None,total_loss=None,total_acc=None):
        if self.batch!=None:
            total_loss+=loss
            try:
                if self.nn.accuracy!=None:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    total_acc+=batch_acc
            except AttributeError:
                pass
            return total_loss,total_acc
        else:
            loss=loss.numpy()
            self.train_loss=loss
            self.train_loss_list.append(loss)
            try:
                if self.nn.accuracy!=None:
                    acc=self.nn.accuracy(output,self.train_labels)
                    acc=acc.numpy()
                    self.train_acc=acc
                    self.train_acc_list.append(acc)
            except AttributeError:
                pass
            if self.test_flag==True:
                if self.process_thread_t==None:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                else:
                    self.test_loss,self.test_acc=self.test(test_batch)
                self.test_loss_list.append(self.test_loss)
                try:
                    if self.nn.accuracy!=None:
                        self.test_acc_list.append(self.test_acc)
                except AttributeError:
                    pass
            return
    
    
    #data_func functin be used for returning batch data and concatenating data.
    def data_func(self,data_batch=None,labels_batch=None,batch=None,index1=None,index2=None,j=None,flag=None):
        if flag==None:
            if batch!=1:
                data_batch=self.train_data[index1:index2]
            else:
                data_batch=self.train_data[j]
            if batch!=1:
                labels_batch=self.train_labels[index1:index2]
            else:
                labels_batch=self.train_labels[j]
        else:
            try:
                data_batch=self.platform.concat([self.train_data[index1:],self.train_data[:index2]],0)
                labels_batch=self.platform.concat([self.train_labels[index1:],self.train_labels[:index2]],0)
            except:
                data_batch=np.concatenate([self.train_data[index1:],self.train_data[:index2]],0)
                labels_batch=np.concatenate([self.train_labels[index1:],self.train_labels[:index2]],0)
        return data_batch,labels_batch
    
    
    #Optimization subfunction,it be used for opt function,it used optimization function of tensorflow platform.
    @function(jit_compile=True)
    def tf_opt(self,data,labels):
        try:  #If neural network object have GradientTape function,kernel will use it or else use other.
            if self.nn.GradientTape!=None:
                tape,output,loss=self.nn.GradientTape(data,labels)
        except AttributeError:
            with self.platform.GradientTape(persistent=True) as tape:
                try:
                    output=self.nn.fp(data)
                    loss=self.nn.loss(output,labels)
                except TypeError:
                    output,loss=self.nn.fp(data,labels)
        try:  #If neural network object have gradient function,kernel will use it otherwise or else use other.
            gradient=self.nn.gradient(tape,loss)
        except AttributeError:
            gradient=tape.gradient(loss,self.nn.param)
        try:
            self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
        except AttributeError:
            self.nn.opt(gradient)
        return output,loss
    
    
    #Optimization subfunction,it be used for opt function,it used optimization function of tensorflow platform and parallel optimization.
    @function(jit_compile=True)
    def tf_opt_t(self,data,labels,t=None,ln=None,u=None):
        try:  #If neural network object have GradientTape function,kernel will use it or else use other.
            if self.nn.GradientTape!=None:
                if type(self.process_thread)==list:
                    tape,output,loss=self.nn.GradientTape(data,labels,u)
                else:
                    tape,output,loss=self.nn.GradientTape(data,labels,t)
        except AttributeError:
            with self.platform.GradientTape(persistent=True) as tape:
                try:
                    try:
                        output=self.nn.fp(data)
                        loss=self.nn.loss(output,labels)
                    except TypeError:
                        output,loss=self.nn.fp(data,labels)
                except TypeError:
                    try:
                        if type(self.process_thread)==list:
                            output=self.nn.fp(data,u)
                        else:
                            output=self.nn.fp(data,t)
                        loss=self.nn.loss(output,labels)
                    except TypeError:
                        if type(self.process_thread)==list:
                            output,loss=self.nn.fp(data,labels,u)
                        else:
                            output,loss=self.nn.fp(data,labels,t)
        try:
            if self.nn.attenuate!=None:
                if type(self.process_thread)==list:
                    self.opt_counter[u]=0
                else:
                    self.opt_counter[t]=0
        except AttributeError:
            pass
        if self.PO==1:
            self.lock[0].acquire()
            try:  #If neural network object have gradient function,kernel will use it otherwise or else use other.
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                gradient=tape.gradient(loss,self.nn.param)
            try:
                if self.nn.attenuate!=None:
                    if type(self.process_thread)==list:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,u)
                    else:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,t)
            except AttributeError:
                pass
            try:
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            except AttributeError:
                try:
                    self.nn.opt(gradient)
                except TypeError:
                    self.nn.opt(gradient,t)
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.lock[0].release()
        elif self.PO==2:
            self.lock[0].acquire()
            try:  #If neural network object have gradient function,kernel will use it otherwise or else use other.
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                gradient=tape.gradient(loss,self.nn.param)
            self.lock[0].release()
            self.lock[1].acquire()
            try:
                if self.nn.attenuate!=None:
                    if type(self.process_thread)==list:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,u)
                    else:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,t)
            except AttributeError:
                pass
            try:
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            except AttributeError:
                try:
                    self.nn.opt(gradient)
                except TypeError:
                    self.nn.opt(gradient,t)
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.lock[1].release()
        return output,loss
    
    
    #Optimization subfunction,it be used for opt function,it used optimization function of pytorch platform.
    def pytorch_opt(self,data,labels):
        output=self.nn.fp(data)
        loss=self.nn.loss(output,labels)
        try:
            self.nn.opt.zero_grad()
            loss.backward()
            self.nn.opt.step()
        except:
            self.nn.opt(loss)
        return output,loss
    
    
    #Main optimization function.
    def opt(self,data,labels):
        try:
            if self.platform.DType!=None:
                output,loss=self.tf_opt(data,labels)
        except AttributeError:
            output,loss=self.pytorch_opt(data,labels)
        return output,loss
    
    
    #Main optimization function,it be used for parallel optimization.
    def opt_t(self,data,labels,t=None,u=None):
        if type(self.process_thread)==list:
            output,loss=self.tf_opt_t(data,labels,u=int(u))
        else:   
            output,loss=self.tf_opt_t(data,labels,int(t))
        return output,loss
    
    
    #Main optimization function,it be used for online training.
    def opt_ol(self,data,labels,t,ln=None):
        try:
            if self.platform.DType!=None:
                if self.process_thread!=None:
                    output,loss=self.tf_opt_t(data,labels,int(t))
                else:
                    output,loss=self.tf_opt(data,labels)
        except AttributeError:
            output,loss=self.pytorch_opt(data,labels)
        return output,loss
    
    
    #Training subfunction,it be used for train function and no parallel training.
    def _train(self,batch=None,_data_batch=None,_labels_batch=None,test_batch=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            if self.train_dataset!=None:
                for data_batch,labels_batch in self.train_dataset:
                    output,batch_loss=self.opt(data_batch,labels_batch)
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc)
            else:
                total_loss=0
                total_acc=0
                batches=int((self.shape0-self.shape0%batch)/batch)
                for j in range(batches):
                    index1=j*batch
                    index2=(j+1)*batch
                    data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,j)
                    output,batch_loss=self.opt(data_batch,labels_batch)
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc)
                    try:
                        self.nn.bc=j
                    except AttributeError:
                        pass
                if self.shape0%batch!=0:
                    batches+=1
                    index1=batches*batch
                    index2=batch-(self.shape0-batches*batch)
                    data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,flag=True)
                    output,batch_loss=self.opt(data_batch,labels_batch)
                    total_loss,total_acc=self.loss_acc(output=output,labels_batch=labels_batch,loss=batch_loss,total_loss=total_loss,total_acc=total_acc)
                    try:
                        self.nn.bc+=1
                    except AttributeError:
                        pass
            try:
                if self.platform.DType!=None:
                    loss=total_loss.numpy()/batches
            except AttributeError:
                loss=total_loss.detach().numpy()/batches
            try:
                if self.nn.accuracy!=None:
                    train_acc=total_acc.numpy()/batches
            except AttributeError:
                pass
            self.train_loss=loss
            self.train_loss_list.append(loss)
            try:
                if self.nn.accuracy!=None:
                    self.train_acc=train_acc
                    self.train_acc_list.append(train_acc)
            except AttributeError:
                pass
            if self.test_flag==True:
                if self.process_thread_t==None:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch)
                else:
                    self.test_loss,self.test_acc=self.test(batch=test_batch)
                self.test_loss_list.append(self.test_loss)
                try:
                    if self.nn.accuracy!=None:
                        self.test_acc_list.append(self.test_acc)
                except AttributeError:
                    pass
        else:
            output,train_loss=self.opt(self.train_data,self.train_labels)
            self.loss_acc(output=output,labels_batch=labels_batch,loss=train_loss,test_batch=test_batch,total_loss=total_loss,total_acc=total_acc)
        return
    
    
    #Training subfunction,it be used for _train_ function and parallel training.
    def train_(self,_data_batch=None,_labels_batch=None,batch=None,batches=None,test_batch=None,index1=None,index2=None,j=None,t=None):
        if batch!=None:
            if index1==batches*batch:
                data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,j,True)
                output,batch_loss=self.opt_t(data_batch,labels_batch,t)
                try:
                    self.nn.bc[t]+=1
                except AttributeError:
                    pass
                try:
                    if self.nn.accuracy!=None:
                        batch_acc=self.nn.accuracy(output,labels_batch)
                        return batch_loss,batch_acc
                except AttributeError:
                    return batch_loss,None
            data_batch,labels_batch=self.data_func(_data_batch,_labels_batch,batch,index1,index2,j)
            output,batch_loss=self.opt_t(data_batch,labels_batch,t)
            try:
                self.nn.bc[t]=j
            except AttributeError:
                pass
            try:
                if self.nn.accuracy!=None:
                    batch_acc=self.nn.accuracy(output,labels_batch)
                    return batch_loss,batch_acc
            except AttributeError:
                return batch_loss,None
        else:
            output,train_loss=self.opt_t(self.train_data,self.train_labels,t)
            if self.PO==1 or self.PO==3:
                self.lock[1].acquire()
                self.total_epoch+=1
                train_loss=train_loss.numpy()
                self.train_loss=train_loss
                self.train_loss_list.append(train_loss)
                try:
                    if self.nn.accuracy!=None:
                        acc=self.nn.accuracy(output,self.train_labels)
                        acc=acc.numpy()
                        self.train_acc=acc
                        self.train_acc_list.append(acc)
                except AttributeError:
                    pass
                if self.test_flag==True:
                    if self.process_thread_t==None:
                        self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch,t)
                    else:
                        self.test_loss,self.test_acc=self.test(batch=test_batch)
                    self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                self.lock[1].release()
            else:
                self.lock[2].acquire()
                self.total_epoch+=1
                train_loss=train_loss.numpy()
                self.train_loss=train_loss
                self.train_loss_list.append(train_loss)
                try:
                    if self.nn.accuracy!=None:
                        acc=self.nn.accuracy(output,self.train_labels)
                        acc=acc.numpy()
                        self.train_acc=acc
                        self.train_acc_list.append(acc)
                except AttributeError:
                    pass
                if self.test_flag==True:
                    if self.process_thread_t==None:
                        self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch,t)
                    else:
                        self.test_loss,self.test_acc=self.test(batch=test_batch)
                    self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                self.lock[2].release()
            return
    
    
    #Training subfunction,it be used for train function and parallel training.
    def _train_(self,batch=None,data_batch=None,labels_batch=None,test_batch=None,t=None):
        total_loss=0
        total_acc=0
        batches=int((self.shape0-self.shape0%batch)/batch)
        if batch!=None:
            for j in range(batches):
                index1=j*batch
                index2=(j+1)*batch
                batch_loss,batch_acc=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
                try:
                    if self.nn.accuracy!=None:
                        total_loss+=batch_loss
                        total_acc+=batch_acc
                except AttributeError:
                    total_loss+=batch_loss
            if self.shape0%batch!=0:
                batches+=1
                index1=batches*batch
                index2=batch-(self.shape0-batches*batch)
                batch_loss,batch_acc=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,None,t)
                try:
                    if self.nn.accuracy!=None:
                        total_loss+=batch_loss
                        total_acc+=batch_acc
                except AttributeError:
                    total_loss+=batch_loss
            loss=total_loss.numpy()/batches
            try:
                if self.nn.accuracy!=None:
                    train_acc=total_acc.numpy()/batches
            except AttributeError:
                pass
            if self.PO==1 or self.PO==3:
                self.lock[1].acquire()
            else:
                self.lock[2].acquire()
            self.total_epoch+=1
            self.train_loss=loss
            self.train_loss_list.append(loss)
            try:
                if self.nn.accuracy!=None:
                    self.train_acc=train_acc
                    self.train_acc_list.append(train_acc)
            except AttributeError:
                pass
            if self.test_flag==True:
                if self.process_thread_t==None:
                    self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch,t)
                else:
                    self.test_loss,self.test_acc=self.test(batch=test_batch)
                self.test_loss_list.append(self.test_loss)
            try:
                if self.nn.accuracy!=None:
                    self.test_acc_list.append(self.test_acc)
            except AttributeError:
                pass
            try:
                self.nn.ec+=1
            except AttributeError:
                pass
            if self.PO==1 or self.PO==3:
                self.lock[1].release()
            else:
                self.lock[2].release()
            return
        else:
            batch_loss,batch_acc=self.train_(data_batch,labels_batch,batch,batches,test_batch,index1,index2,j,t)
            return
    
    
    #Training subfunction,it be used for train function and parallel training.
    def train7(self,train_ds,t,test_batch):
        if type(self.process_thread)==list:
            if self.PO==1 or self.PO==3:
                self.lock[1].acquire()
            else:
                self.lock[2].acquire()
            u=self.thread_num[t].pop(0)
            if self.PO==1 or self.PO==3:
                self.lock[1].release()
            else:
                self.lock[2].release()
        while True:
            for data_batch,labels_batch in train_ds:
                if type(self.process_thread)==list:
                    output,batch_loss=self.opt_t(data_batch,labels_batch,u=u)
                else:
                    output,batch_loss=self.opt_t(data_batch,labels_batch,t)
                try:
                    if type(self.process_thread)==list:
                       self.nn.bc[u]+=1 
                    else:
                        self.nn.bc[t]+=1
                except AttributeError:
                    pass
                try:
                    if self.nn.accuracy!=None:
                        batch_acc=self.nn.accuracy(output,labels_batch)
                except AttributeError:
                    pass
                try:
                    if self.nn.accuracy!=None:
                        if type(self.process_thread)==list:
                            self.total_loss[u]+=batch_loss
                            self.total_acc[u]+=batch_acc
                        else:
                            self.total_loss[t]+=batch_loss
                            self.total_acc[t]+=batch_acc
                except AttributeError:
                    if type(self.process_thread)==list:
                        self.total_loss[u]+=batch_loss
                    else:
                        self.total_loss[t]+=batch_loss
                if type(self.process_thread)==list:
                    self.batch_counter[u]+=1
                else:
                    self.batch_counter[t]+=1
                if self.PO==1 or self.PO==3:
                    self.lock[1].acquire()
                else:
                    self.lock[2].acquire()
                batches=np.sum(self.batch_counter)
                if batches>=self.batches:
                    self.batch_counter=self.batch_counter*0
                    loss=np.sum(self.total_loss)/batches
                    try:
                        if self.nn.accuracy!=None:
                            train_acc=np.sum(self.total_acc)/batches
                    except AttributeError:
                        pass
                    self.total_epoch+=1
                    self.train_loss=loss
                    self.train_loss_list.append(loss)
                    try:
                        if self.nn.accuracy!=None:
                            self.train_acc=train_acc
                            self.train_acc_list.append(train_acc)
                    except AttributeError:
                        pass
                    if self.test_flag==True:
                        if self.process_thread_t==None:
                            self.test_loss,self.test_acc=self.test(self.test_data,self.test_labels,test_batch,t)
                        else:
                            self.test_loss,self.test_acc=self.test(batch=test_batch)
                        self.test_loss_list.append(self.test_loss)
                    try:
                        if self.nn.accuracy!=None:
                            self.test_acc_list.append(self.test_acc)
                    except AttributeError:
                        pass
                    self.epoch_counter+=1
                    try:
                        if type(self.process_thread)==list:
                           self.nn.bc[u]=0
                        else:
                            self.nn.bc[t]=0
                    except AttributeError:
                        pass
                    try:
                        self.nn.ec+=1
                    except AttributeError:
                        pass
                    self.total_loss=self.total_loss*0
                    try:
                        if self.nn.accuracy!=None:
                            self.total_acc=self.total_acc*0
                    except AttributeError:
                        pass
                if self.PO==1 or self.PO==3:
                    self.lock[1].release()
                else:
                    self.lock[2].release()
                if self.epoch_counter==self.epoch_:
                    return
    
    
    #Main training function.
    def train(self,batch=None,epoch=None,test_batch=None,save=None,one=True,p=None,s=None):
        if self.process_thread!=None:
            if self.PO==1 or self.PO==3:
                self.lock[1].acquire()
            else:
                self.lock[2].acquire()
            t=self.process_thread_num.pop(0)
            self.process_thread_counter+=1
            if self.epoch_!=None:
                if self.train_dataset==None:
                    if t==0:
                        if self.batches==None:
                            self.batches=int((self.shape0-self.shape0%batch)/batch)
                            if self.shape0%batch!=0:
                                self.batches+=1
            if self.PO==1 or self.PO==3:
                self.lock[1].release()
            else:
                self.lock[2].release()
        self.batch=batch
        self.epoch=0
        self.train_counter+=1
        if self.epoch_!=None:
            if self.train_dataset!=None:
                train_ds=self.train_dataset
            else:
                if self.data_segment_flag==True:
                    train_ds=self.platform.data.Dataset.from_tensor_slices((self.train_data[t],self.train_labels[t])).batch(batch)
                elif self.buffer_size!=None:
                    train_ds=self.platform.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).shuffle(self.buffer_size).batch(batch)
                else:
                    train_ds=self.platform.data.Dataset.from_tensor_slices((self.train_data,self.train_labels)).batch(batch)
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if s==None:
            self.s=1
            self.file_list=None
        else:
            self.s=s-1
            self.file_list=[]
        data_batch=None
        labels_batch=None
        if self.process_thread!=None and self.epoch_!=None:
            if self.multiprocessing_threading==None:
                self.train7(train_ds,t,test_batch)
            else:
                train7=self.train7
                class thread(self.multiprocessing_threading.Thread):     
                    def run(self):              
                        train7(train_ds,t,test_batch)
                for _ in range(self.process_thread[1]):
                    _thread=thread()
                    _thread.start()
                for _ in range(self.process_thread[1]):
                    _thread.join()
        elif epoch!=None:
            for i in range(epoch):
                t1=time.time()
                if self.process_thread==None:
                    self._train(batch,data_batch,labels_batch,test_batch)
                else:
                    self._train_(batch,data_batch,labels_batch,test_batch,t)
                if self.process_thread==None:
                    try:
                        self.nn.ec+=1
                    except AttributeError:
                        pass
                if self.process_thread==None:
                    self.total_epoch+=1
                if self.process_thread==None:
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
                    if i%p==0:
                        if self.test_flag==False:
                            try:
                                if self.nn.accuracy!=None:
                                    print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))
                                    print()
                            except AttributeError:
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                print()
                        else:
                            try:
                                if self.nn.accuracy!=None:
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc))
                                    print()
                            except AttributeError:   
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                print()
                    if save!=None and i%s==0:
                        self.save(self.total_epoch,one)
                t2=time.time()
                if self.process_thread==None:
                    self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                if self.process_thread==None:
                    self._train(test_batch=test_batch)
                else:
                    self._train_(test_batch=test_batch,t=t)
                i+=1
                if self.process_thread==None:
                    try:
                        self.nn.ec+=1
                    except AttributeError:
                        pass
                if self.process_thread==None:
                    self.total_epoch+=1
                if self.process_thread==None:
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
                    if i%p==0:
                        if self.test_flag==False:
                            try:
                                if self.nn.accuracy!=None:
                                    print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f}'.format(i+1,self.train_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.6f}'.format(i+1,self.train_acc))
                                    print()
                            except AttributeError:
                                print('epoch:{0}   loss:{1:.6f}'.format(i+1,self.train_loss))
                                print()
                        else:
                            try:
                                if self.nn.accuracy!=None:
                                    print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                    if self.acc_flag=='%':
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc*100,self.test_acc*100))
                                    else:
                                        print('epoch:{0}   accuracy:{1:.1f},test accuracy:{2:.1f}'.format(i+1,self.train_acc,self.test_acc))
                                    print()
                            except AttributeError:   
                                print('epoch:{0}   loss:{1:.6f},test loss:{2:.6f}'.format(i+1,self.train_loss,self.test_loss))
                                print()
                    if save!=None and i%s==0:
                        self.save(self.total_epoch,one)
                t2=time.time()
                if self.process_thread==None:
                    self.time+=(t2-t1)
        if save!=None:
            self.save()
        if self.process_thread==None:
            self._time=self.time-int(self.time)
            if self._time<0.5:
                self.time=int(self.time)
            else:
                self.time=int(self.time)+1
            self.total_time+=self.time
        if self.process_thread==None:
            if self.test_flag==False:
                print('last loss:{0:.6f}'.format(self.train_loss))
            else:
                print('last loss:{0:.6f},last test loss:{1:.6f}'.format(self.train_loss,self.test_loss))
            try:
                if self.nn.accuracy!=None:
                    if self.acc_flag=='%':
                        if self.test_flag==False:
                            print('last accuracy:{0:.1f}'.format(self.train_acc*100))
                        else:
                            print('last accuracy:{0:.1f},last test accuracy:{1:.1f}'.format(self.train_acc*100,self.test_acc*100))
                    else:
                        if self.test_flag==False:
                            print('last accuracy:{0:.6f}'.format(self.train_acc))
                        else:
                            print('last accuracy:{0:.6f},last test accuracy:{1:.6f}'.format(self.train_acc,self.test_acc))   
            except AttributeError:
                pass
            print()
            print('time:{0}s'.format(self.time))
        if self.process_thread==None:
            self.training_flag=False
        if self.process_thread!=None:
            if self.PO==1 or self.PO==3:
                self.lock[1].acquire()
            else:
                self.lock[2].acquire()
            self.process_thread_counter-=1
            if self.PO==1 or self.PO==3:
                self.lock[1].release()
            else:
                self.lock[2].release()
        return
    
    
    def train_ol(self):
        if self.process_thread!=None:
            if self.PO==1 or self.PO==3:
                self.lock[1].acquire()
            else:
                self.lock[2].acquire()
            t=self.process_thread_num.pop(0)
            self.process_thread_counter+=1
            if self.PO==1 or self.PO==3:
                self.lock[1].release()
            else:
                self.lock[2].release()
        while True:
            if self.process_thread!=None:
                if self.save_flag==True:
                    if self.PO==1 or self.PO==3:
                        self.lock[1].acquire()
                    else:
                        self.lock[2].acquire()
                    self.save()
                    if self.PO==1 or self.PO==3:
                        self.lock[1].release()
                    else:
                        self.lock[2].release()
                try:
                    data=self.nn.ol(t)
                except:
                    self.exception_list[t]=True
                    continue
                try:
                    output,loss=self.opt_ol(data[0],data[1],t)
                except:
                    continue
                loss=loss.numpy()
                if self.PO==1 or self.PO==3:
                    self.lock[1].acquire()
                else:
                    self.lock[2].acquire()
                if len(self.nn.train_loss_list)==self.nn.max_length:
                    del self.nn.train_loss_list[0]
                self.nn.train_loss_list.append(loss)
                try:
                    if self.nn.accuracy!=None:
                        try:
                            train_acc=self.nn.accuracy(output,data[1])
                        except:
                            continue
                        if len(self.nn.train_acc_list)==self.nn.max_length:
                            del self.nn.train_acc_list[0]
                        self.train_acc_list.append(train_acc)
                except AttributeError:
                    pass
                try:
                    self.nn.c+=1
                except AttributeError:
                    pass
                if self.PO==1 or self.PO==3:
                    self.lock[1].release()
                else:
                    self.lock[2].release()
            else:
                if self.save_flag==True:
                    self.save()
                data=self.nn.ol()
                if data=='stop':
                    return
                output,loss=self.opt(data[0],data[1])
                loss=loss.numpy()
                if len(self.nn.train_loss_list)==self.nn.max_length:
                    del self.nn.train_loss_list[0]
                self.nn.train_loss_list.append(loss)
                try:
                    if self.nn.accuracy!=None:
                        train_acc=self.nn.accuracy(output,data[1])
                        if len(self.nn.train_acc_list)==self.nn.max_length:
                            del self.nn.train_acc_list[0]
                        self.train_acc_list.append(train_acc)
                except AttributeError:
                    pass
                try:
                    self.nn.c+=1
                except AttributeError:
                    pass
        if self.process_thread!=None:
            if self.PO==1 or self.PO==3:
                self.lock[1].acquire()
            else:
                self.lock[2].acquire()
            self.process_thread_counter-=1
            if self.PO==1 or self.PO==3:
                self.lock[1].release()
            else:
                self.lock[2].release()
        return
    
    
    def test(self,test_data=None,test_labels=None,batch=None,t=None):
        if type(test_data)==list:
            data_batch=[x for x in range(len(test_data))]
        if type(test_labels)==list:
            labels_batch=[x for x in range(len(test_labels))]
        if batch!=None:
            total_loss=0
            total_acc=0
            if self.test_dataset!=None:
                for data_batch,labels_batch in self.test_dataset:
                    if self.process_thread==None or t==None:
                        output=self.nn.fp(data_batch)
                    else:
                        output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                    total_loss+=batch_loss
                    try:
                        if self.nn.accuracy!=None:
                            batch_acc=self.nn.accuracy(output,labels_batch)
                            total_acc+=batch_acc
                    except AttributeError:
                        pass
            else:
                total_loss=0
                total_acc=0
                if type(test_data)==list:
                    batches=int((test_data[0].shape[0]-test_data[0].shape[0]%batch)/batch)
                    shape0=test_data[0].shape[0]
                else:
                    batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
                    shape0=test_data.shape[0]
                for j in range(batches):
                    index1=j*batch
                    index2=(j+1)*batch
                    if type(test_data)==list:
                        for i in range(len(test_data)):
                            data_batch[i]=test_data[i][index1:index2]
                    else:
                        data_batch=test_data[index1:index2]
                    if type(test_labels)==list:
                        for i in range(len(test_labels)):
                            labels_batch[i]=test_labels[i][index1:index2]
                    else:
                        labels_batch=test_labels[index1:index2]
                    if self.process_thread==None or t==None:
                        output=self.nn.fp(data_batch)
                    else:
                        output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                    total_loss+=batch_loss
                    try:
                        if self.nn.accuracy!=None:
                            batch_acc=self.nn.accuracy(output,labels_batch)
                            total_acc+=batch_acc
                    except AttributeError:
                        pass
                if shape0%batch!=0:
                    batches+=1
                    index1=batches*batch
                    index2=batch-(shape0-batches*batch)
                    try:
                        if type(test_data)==list:
                            for i in range(len(test_data)):
                                data_batch[i]=self.platform.concat([test_data[i][index1:],test_data[i][:index2]],0)
                        else:
                            data_batch=self.platform.concat([test_data[index1:],test_data[:index2]],0)
                        if type(self.test_labels)==list:
                            for i in range(len(test_labels)):
                                labels_batch[i]=self.platform.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                        else:
                            labels_batch=self.platform.concat([test_labels[index1:],test_labels[:index2]],0)
                    except:
                        if type(test_data)==list:
                            for i in range(len(test_data)):
                                data_batch[i]=self.platform.concat([test_data[i][index1:],test_data[i][:index2]],0)
                        else:
                            data_batch=self.platform.concat([test_data[index1:],test_data[:index2]],0)
                        if type(self.test_labels)==list:
                            for i in range(len(test_labels)):
                                labels_batch[i]=self.platform.concat([test_labels[i][index1:],test_labels[i][:index2]],0)
                        else:
                            labels_batch=self.platform.concat([test_labels[index1:],test_labels[:index2]],0)
                    if self.process_thread==None or t==None:
                        output=self.nn.fp(data_batch)
                    else:
                        output=self.nn.fp(data_batch,t)
                    batch_loss=self.nn.loss(output,labels_batch)
                    total_loss+=batch_loss
                    try:
                        if self.nn.accuracy!=None:
                            batch_acc=self.nn.accuracy(output,labels_batch)
                            total_acc+=batch_acc
                    except AttributeError:
                        pass
            test_loss=total_loss.numpy()/batches
            try:
                if self.nn.accuracy!=None:
                    test_acc=total_acc.numpy()/batches
            except AttributeError:
                pass
        else:
            if self.process_thread==None or t==None:
                output=self.nn.fp(test_data)
            else:
                output=self.nn.fp(test_data,t)
            test_loss=self.nn.loss(output,test_labels)
            test_loss=test_loss.numpy()
            try:
                if self.nn.accuracy!=None:
                    test_acc=self.nn.accuracy(output,test_labels)
                    test_acc=test_acc.numpy()
            except AttributeError:
                pass
        try:
            if self.nn.accuracy!=None:
                return test_loss,test_acc
        except AttributeError:
            return test_loss,None


    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('train loss:{0:.6f}'.format(self.train_loss))
        try:
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(self.total_epoch),self.train_acc_list)
                plt.title('train acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                if self.acc_flag=='%':
                    print('train acc:{0:.1f}'.format(self.train_acc*100))
                else:
                    print('train acc:{0:.6f}'.format(self.train_acc)) 
        except AttributeError:
            pass
        return
    
    
    def visualize_test(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('test loss:{0:.6f}'.format(self.test_loss))
        try:
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(self.total_epoch),self.test_acc_list)
                plt.title('test acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                if self.acc_flag=='%':
                    print('test acc:{0:.1f}'.format(self.test_acc*100))
                else:
                    print('test acc:{0:.6f}'.format(self.test_acc))  
        except AttributeError:
            pass
        return 
    
    
    def comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch),self.test_loss_list,'r-',label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('train loss:{0}'.format(self.train_loss))
        plt.legend()
        try:
            if self.nn.accuracy!=None:
                plt.figure(2)
                plt.plot(np.arange(self.total_epoch),self.train_acc_list,'b-',label='train acc')
                if self.test_flag==True:
                    plt.plot(np.arange(self.total_epoch),self.test_acc_list,'r-',label='test acc')
                plt.xlabel('epoch')
                plt.ylabel('acc')
                plt.legend()
                if self.acc_flag=='%':
                    print('train acc:{0:.1f}'.format(self.train_acc*100))
                else:
                    print('train acc:{0:.6f}'.format(self.train_acc))
        except AttributeError:
            pass
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            if self.acc_flag=='%':
                print('test acc:{0:.1f}'.format(self.test_acc*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc)) 
        return
    
    
    def save_p(self):
        parameter_file=open('param.dat','wb')
        pickle.dump(self.nn.param,parameter_file)
        parameter_file.close()
        return
    
    
    def save(self,i=None,one=True):
        if one==True:
            output_file=open(self.filename,'wb')
        else:
            filename=self.filename.replace(self.filename[self.filename.find('.'):],'-{0}.dat'.format(i))
            output_file=open(filename,'wb')
            self.file_list.append([filename])
            if len(self.file_list)>self.s+1:
                os.remove(self.file_list[0][0])
                del self.file_list[0]
        try:
            pickle.dump(self.nn,output_file)
        except:
            opt=self.nn.opt
            self.nn.opt=None
            pickle.dump(self.nn,output_file)
            self.nn.opt=opt
        try:
            pickle.dump(self.platform.keras.optimizers.serialize(opt),output_file)
        except:
            pickle.dump(self.nn.serialize(),output_file)
        else:
            pickle.dump(None,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.acc_flag,output_file)
        pickle.dump(self.file_list,output_file)
        pickle.dump(self.train_counter,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_acc,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_acc_list,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_acc,output_file)
            pickle.dump(self.test_loss_list,output_file)
            pickle.dump(self.test_acc_list,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
	
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        opt_serialized=pickle.load(input_file)
        try:
            self.nn.opt=self.platform.keras.optimizers.deserialize(opt_serialized)
        except:
            self.nn.deserialize(opt_serialized)
        else:
            pass
        self.batch=pickle.load(input_file)
        self.acc_flag=pickle.load(input_file)
        self.file_list=pickle.load(input_file)
        self.train_counter=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_acc=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_acc_list=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_acc=pickle.load(input_file)
            self.test_loss_list=pickle.load(input_file)
            self.test_acc_list=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
