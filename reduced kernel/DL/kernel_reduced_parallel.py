from tensorflow import function
import numpy as np
import matplotlib.pyplot as plt


#Keep only parallel part.
#You can analyze kernel by example.
'''
multithreading example:
import kernel_reduced_parallel as k   #import kernel
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
import kernel_reduced_parallel as k   #import kernel
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
        self.batch=None
        self.epoch=0
        self.acc_flag='%'
        self.train_counter=0
        self.opt_counter=None
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
    def train(self,batch=None,epoch=None,test_batch=None):
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
                self._train_(batch,data_batch,labels_batch,test_batch,t)
        else:
            while True:
                self._train_(test_batch=test_batch,t=t)
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
