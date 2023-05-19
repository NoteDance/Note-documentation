import tensorflow as tf
from tensorflow import data as tf_data
import numpy as np
import matplotlib.pyplot as plt
import statistics


#You can analyze kernel by example.
'''
multithreading example:
import kernel_reduced as k   #import kernel
import DQN as d
import threading
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=1                    #use PO1
kernel.multiprocessing_threading=threading
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(5):
	_thread=thread()
	_thread.start()
for _ in range(5):
	_thread.join()
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
'''
class kernel:
    def __init__(self,nn=None,process_thread=None,save_episode=False):
        self.nn=nn
        if process_thread!=None:
            self.process_thread_num=np.arange(process_thread)
            self.process_thread_num=list(self.process_thread_num)
            self.reward=np.zeros(process_thread,dtype=np.float32)
            self.loss=np.zeros(process_thread,dtype=np.float32)
            self.sc=np.zeros(process_thread,dtype=np.float32)
            self.opt_counter=np.zeros(process_thread,dtype=np.float32)
        self.multiprocessing_threading=None
        self.state_pool={}
        self.action_pool={}
        self.next_state_pool={}
        self.reward_pool={}
        self.done_pool={}
        self.episode_set=[]
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.episode_=None
        self.episode=0
        self.update_step=None
        self.trial_count=None
        self.process_thread=process_thread
        self.process_thread_counter=0
        self.lock=None
        self.pool_lock=[]
        self.probability_list=[]
        self.running_flag_list=[]
        self.finish_list=[]
        self.running_flag=np.array(0,dtype=np.int8)
        self.PN=True
        self.PO=None
        self.max_episode_count=None
        self.save_episode=save_episode
        self.filename='save.dat'
        self.reward_list=[]
        self.loss_list=[]
        self.total_episode=0
        self.total_time=0
    
    
    def action_vec(self):
        self.action_one=np.ones(self.action_count,dtype=np.int8)
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None,end_loss=None):
        if epsilon!=None:
            self.epsilon=np.ones(self.process_thread)*epsilon
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_step!=None:
            self.update_step=update_step
        if trial_count!=None:
            self.trial_count=trial_count
        if criterion!=None:
            self.criterion=criterion
        if end_loss!=None:
            self.end_loss=end_loss
        if epsilon!=None:
            self.action_vec()
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon):
        action_prob=self.action_one*epsilon/len(self.action_one)
        best_a=np.argmax(self.nn.nn.fp(s))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done,t,index):
        if self.PN==True:
            self.pool_lock[index].acquire()
            try:
                if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None:
                    self.state_pool[index]=s
                    if type(a)==int:
                        a=np.array(a)
                        self.action_pool[index]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[index]=a
                    self.next_state_pool[index]=np.expand_dims(next_s,axis=0)
                    self.reward_pool[index]=np.expand_dims(r,axis=0)
                    self.done_pool[index]=np.expand_dims(done,axis=0)
                else:
                    try:
                        self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)
                        if type(a)==int:
                            a=np.array(a,np.int64)
                            self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                        else:
                            self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                        self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)
                        self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                        self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                    except:
                        pass
                if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size:
                    self.state_pool[index]=self.state_pool[index][1:]
                    self.action_pool[index]=self.action_pool[index][1:]
                    self.next_state_pool[index]=self.next_state_pool[index][1:]
                    self.reward_pool[index]=self.reward_pool[index][1:]
                    self.done_pool[index]=self.done_pool[index][1:]
            except:
                self.pool_lock[index].release()
                return
            self.pool_lock[index].release()
        else:
            if type(self.state_pool[t])==np.ndarray and self.state_pool[t]==None:
                self.state_pool[t]=s
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool[t]=np.expand_dims(a,axis=0)
                else:
                    self.action_pool[t]=a
                self.next_state_pool[t]=np.expand_dims(next_s,axis=0)
                self.reward_pool[t]=np.expand_dims(r,axis=0)
                self.done_pool[t]=np.expand_dims(done,axis=0)
            else:
                self.state_pool[t]=np.concatenate((self.state_pool[t],s),0)
                if type(a)==int:
                    a=np.array(a,np.int64)
                    self.action_pool[t]=np.concatenate((self.action_pool[t],np.expand_dims(a,axis=0)),0)
                else:
                    self.action_pool[t]=np.concatenate((self.action_pool[t],a),0)
                self.next_state_pool[t]=np.concatenate((self.next_state_pool[t],np.expand_dims(next_s,axis=0)),0)
                self.reward_pool[t]=np.concatenate((self.reward_pool[t],np.expand_dims(r,axis=0)),0)
                self.done_pool[t]=np.concatenate((self.done_pool[t],np.expand_dims(done,axis=0)),0)
            if type(self.state_pool[t])==np.ndarray and len(self.state_pool[t])>self.pool_size:
                self.state_pool[t]=self.state_pool[t][1:]
                self.action_pool[t]=self.action_pool[t][1:]
                self.next_state_pool[t]=self.next_state_pool[t][1:]
                self.reward_pool[t]=self.reward_pool[t][1:]
                self.done_pool[t]=self.done_pool[t][1:]
        return
    
    
    def get_index(self,t):
        if self.PN==True:
            while len(self.running_flag_list)<t:
                pass
            if len(self.running_flag_list)==t:
                if self.PO==1 or self.PO==3:
                    self.lock[2].acquire()
                else:
                    self.lock[3].acquire()
                self.running_flag_list.append(self.running_flag[1:].copy())
                if self.PO==1 or self.PO==3:
                    self.lock[2].release()
                else:
                    self.lock[3].release()
            if len(self.running_flag_list[t])<self.process_thread_counter or np.sum(self.running_flag_list[t])>self.process_thread_counter:
                self.running_flag_list[t]=self.running_flag[1:].copy()
            while len(self.probability_list)<t:
                pass
            if len(self.probability_list)==t:
                if self.PO==1 or self.PO==3:
                    self.lock[2].acquire()
                else:
                    self.lock[3].acquire()
                self.probability_list.append(np.array(self.running_flag_list[t],dtype=np.float16)/np.sum(self.running_flag_list[t]))
                if self.PO==1 or self.PO==3:
                    self.lock[2].release()
                else:
                    self.lock[3].release()
            self.probability_list[t]=np.array(self.running_flag_list[t],dtype=np.float16)/np.sum(self.running_flag_list[t])
            while True:
                index=np.random.choice(len(self.probability_list[t]),p=self.probability_list[t])
                if index in self.finish_list:
                    continue
                else:
                    break
        else:
            index=None
        return index
    
    
    def env(self,s,epsilon,t):
        try:
            if self.nn.nn!=None:
                s=np.expand_dims(s,axis=0)
                action_prob=self.epsilon_greedy_policy(s,epsilon)
                a=np.random.choice(self.action_count,p=action_prob)
                next_s,r,done=self.nn.env(a,t)
        except AttributeError:
            try:
                if self.nn.action!=None:
                    s=np.expand_dims(s,axis=0)
                    a=self.nn.action(s).numpy()
            except AttributeError:
                s=np.expand_dims(s,axis=0)
                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                next_s,r,done=self.nn.env(a,t)
        index=self.get_index(t)
        r=np.array(r,dtype=np.float32)
        done=np.array(done,dtype=np.float32)
        self.pool(s,a,next_s,r,done,t,index)
        if self.save_episode==True:
            episode=[s,a,next_s,r]
            return next_s,r,done,episode,index
        else:
            return next_s,r,done,None,index
    
    
    def end(self):
        if self.trial_count!=None:
            if len(self.reward_list)>=self.trial_count:
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                if self.criterion!=None and avg_reward>=self.criterion:
                    return True
    
    
    @tf.function(jit_compile=True)
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,t,ln=None):
        with tf.GradientTape(persistent=True) as tape:
            try:
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            except TypeError:
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
        try:
            if self.nn.attenuate!=None:
                self.opt_counter[t]=0
        except AttributeError:
            pass
        if self.PO==1:
            self.lock[0].acquire()
            try:
                gradient=self.nn.gradient(tape,loss)
                try:
                    if self.nn.attenuate!=None:
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
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                        try:
                            if self.nn.attenuate!=None:
                                gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                        except AttributeError:
                            pass
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                        actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                        critic_gradient=tape.gradient(loss[1],self.nn.param[1])
                        try:
                            if self.nn.attenuate!=None:
                                actor_gradient=self.nn.attenuate(actor_gradient,self.opt_counter,t)
                                critic_gradient=self.nn.attenuate(critic_gradient,self.opt_counter,t)
                        except AttributeError:
                            pass
                        self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                        self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.lock[0].release()
        elif self.PO==2:
            self.lock[0].acquire()
            try:
                gradient=self.nn.gradient(tape,loss)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        gradient=tape.gradient(loss,self.nn.param)
                except AttributeError:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
            self.lock[0].release()
            self.lock[1].acquire()
            try:
                if self.nn.attenuate!=None:
                    try:
                        gradient=self.nn.attenuate(gradient,self.opt_counter,t)
                    except NameError:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.opt_counter,t)
                        critic_gradient=self.nn.attenuate(critic_gradient,self.opt_counter,t)
            except AttributeError:
                pass
            try:
                if self.nn.gradient!=None:
                    try:
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                    except AttributeError:
                        try:
                            self.nn.opt(gradient)
                        except TypeError:
                            self.nn.opt(gradient,t)
            except AttributeError:
                try:
                    if self.nn.nn!=None:
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                except AttributeError:
                    self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                    self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
            try:
                if self.nn.attenuate!=None:
                    self.opt_counter+=1
            except AttributeError:
                pass
            self.lock[1].release()
        return loss
    
    
    def _train(self,t,j=None,batches=None,length=None,ln=None):
        if j==batches-1:
            index1=batches*self.batch
            index2=self.batch-(length-batches*self.batch)
            state_batch=np.concatenate((self.state_pool[t][index1:length],self.state_pool[t][:index2]),0)
            action_batch=np.concatenate((self.action_pool[t][index1:length],self.action_pool[t][:index2]),0)
            next_state_batch=np.concatenate((self.next_state_pool[t][index1:length],self.next_state_pool[t][:index2]),0)
            reward_batch=np.concatenate((self.reward_pool[t][index1:length],self.reward_pool[t][:index2]),0)
            done_batch=np.concatenate((self.done_pool[t][index1:length],self.done_pool[t][:index2]),0)
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
            self.loss[t]+=loss
            try:
                self.nn.bc[t]+=1
            except AttributeError:
                pass
        else:
            index1=j*self.batch
            index2=(j+1)*self.batch
            state_batch=self.state_pool[t][index1:index2]
            action_batch=self.action_pool[t][index1:index2]
            next_state_batch=self.next_state_pool[t][index1:index2]
            reward_batch=self.reward_pool[t][index1:index2]
            done_batch=self.done_pool[t][index1:index2]
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
            self.loss[t]+=loss
            try:
                self.nn.bc[t]=j
            except AttributeError:
                pass
        return
    
    
    def train_(self,t,ln=None):
        train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool[t],self.action_pool[t],self.next_state_pool[t],self.reward_pool[t],self.done_pool[t])).shuffle(len(self.state_pool[t])).batch(self.batch)
        for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
            self.suspend_func(t)
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,t)
            self.loss[t]+=loss
            try:
                self.nn.bc[t]+=1
            except AttributeError:
                pass
        return
            
    
    def _train_(self,t):
        if len(self.done_pool[t])<self.batch:
            return
        else:
            self.loss[t]=0
            if self.PN==True:
                length=len(self.done_pool[t])
                batches=int((length-length%self.batch)/self.batch)
                if length%self.batch!=0:
                    batches+=1
                for j in range(batches):
                    self._train(t,j,batches,length)
            else:
                try:
                    self.nn.bc[t]=0
                except AttributeError:
                    pass
                self.train_(t)
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.lock[1].acquire()
                else:
                    self.lock[2].acquire()
            else:
                self.lock[1].acquire()
            if self.update_step!=None:
                if self.sc[t]%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.lock[1].release()
                else:
                    self.lock[2].release()
            else:
                self.lock[1].release()
            self.loss[t]=self.loss[t]/batches
        self.sc[t]+=1
        try:
            self.nn.ec[t]+=1
        except AttributeError:
            pass
        return
    
    
    def train(self,episode_count):
        if self.PN==True:
            if self.PO==1 or self.PO==3:
                self.lock[2].acquire()
            else:
                self.lock[3].acquire()
        else:
            self.lock[0].acquire()
        t=self.process_thread_num.pop(0)
        t=int(t)
        self.state_pool[t]=None
        self.action_pool[t]=None
        self.next_state_pool[t]=None
        self.reward_pool[t]=None
        self.done_pool[t]=None
        self.running_flag=np.append(self.running_flag,np.array(1,dtype=np.int8))
        if self.multiprocessing_threading!=None:
            self.pool_lock.append(self.multiprocessing_threading.Lock())
        self.process_thread_counter+=1
        self.finish_list.append(None)
        try:
            epsilon=self.epsilon[t]
        except:
            epsilon=None
        try:
            self.nn.ec.append(0)
        except AttributeError:
            pass
        try:
            self.nn.bc.append(0)
        except AttributeError:
            pass
        if self.PN==True:
            if self.PO==1 or self.PO==3:
                self.lock[2].release()
            else:
                self.lock[3].release()
        else:
            self.lock[0].release()
        for k in range(episode_count):
            episode=[]
            s=self.nn.env(t=t,initial=True)
            if self.episode_step==None:
                while True:
                    next_s,r,done,_episode,index=self.env(s,epsilon,t)
                    self.reward[t]+=r
                    s=next_s
                    if type(self.done_pool[t])==np.ndarray:
                        self._train_(t)
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.lock[2].acquire()
                            else:
                                self.lock[3].acquire()
                        else:
                            self.lock[0].acquire()
                        self.total_episode+=1
                        self.loss_list.append(self.loss[t])
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.lock[2].release()
                            else:
                                self.lock[3].release()
                        else:
                            self.lock[0].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
            else:
                for l in range(self.episode_step):
                    next_s,r,done,_episode,index=self.env(s,epsilon,t)
                    self.reward[t]+=r
                    s=next_s
                    if type(self.done_pool[t])==np.ndarray:
                        self._train_(t)
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.lock[2].acquire()
                            else:
                                self.lock[3].acquire()
                        else:
                            self.lock[0].acquire()
                        self.total_episode+=1
                        self.loss_list.append(self.loss[t])
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.lock[2].release()
                            else:
                                self.lock[3].release()
                        else:
                            self.lock[0].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
                    if l==self.episode_step-1:
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.lock[2].acquire()
                            else:
                                self.lock[3].acquire()
                        else:
                            self.lock[0].acquire()
                        self.total_episode+=1
                        self.loss_list.append(self.loss[t])
                        if self.PN==True:
                            if self.PO==1 or self.PO==3:
                                self.lock[2].release()
                            else:
                                self.lock[3].release()
                        else:
                            self.lock[0].release()
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.lock[2].acquire()
                else:
                    self.lock[3].acquire()
            else:
                self.lock[0].acquire()
            self.reward_list.append(self.reward[t])
            self.reward[t]=0
            if self.save_episode==True:
                self.episode_set.append(episode)
                if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count:
                    self.save_episode=False
            if self.PN==True:
                if self.PO==1 or self.PO==3:
                    self.lock[2].release()
                else:
                    self.lock[3].release()
            else:
                self.lock[0].release()
        try:
            if self.nn.row!=None:
                pass
        except AttributeError:
            self.running_flag[t+1]=0
        if self.PO==1 or self.PO==3:
            self.lock[2].acquire()
        else:
            self.lock[3].acquire()
        self.process_thread_counter-=1
        if t not in self.finish_list:
            self.finish_list[t]=t
        if self.PO==1 or self.PO==3:
            self.lock[2].release()
        else:
            self.lock[3].release()
        del self.state_pool[t]
        del self.action_pool[t]
        del self.next_state_pool[t]
        del self.reward_pool[t]
        del self.done_pool[t]
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_list)),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.loss_list)),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def visualize_reward_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_list)),self.reward_list,'r-',label='reward')
        plt.plot(np.arange(len(self.loss_list)),self.loss_list,'b-',label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('reward and loss')
        return
