import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Value,Array
import numpy as np
import matplotlib.pyplot as plt
import statistics


class kernel:
    def __init__(self,nn=None,process=None):
        self.nn=nn # the neural network model
        if process!=None: # the number of processes
            self.reward=np.zeros(process,dtype=np.float32) # the reward array for each process
            self.sc=np.zeros(process,dtype=np.int32) # the step counter array for each process
        self.state_pool={} # the dictionary to store the state pool for each process
        self.action_pool={} # the dictionary to store the action pool for each process
        self.next_state_pool={} # the dictionary to store the next state pool for each process
        self.reward_pool={} # the dictionary to store the reward pool for each process
        self.done_pool={} # the dictionary to store the done flag pool for each process
        self.epsilon=None # the epsilon value for epsilon-greedy policy
        self.episode_step=None # the maximum number of steps per episode
        self.pool_size=None # the maximum size of the experience pool
        self.batch=None # the batch size for training
        self.episode=0 # the episode counter
        self.update_step=None # the frequency of updating the target network parameters
        self.trial_count=None # the number of trials to calculate the average reward
        self.process=process # the number of processes
        self.process_counter=0 # the counter of running processes
        self.probability_list=[] # the list to store the probability distribution of running processes
        self.running_flag_list=[] # the list to store the running flag of each process
        self.finish_list=[] # the list to store the finished processes
        self.running_flag=[] # the list to store the running flag of all processes (including 0)
        self.PO=None # the optimization strategy (1: lock, 2: global lock, 3: no lock)
        self.priority_flag=False # whether to use priority optimization or not
        self.priority_p=0 # the priority process index for optimization
        self.max_opt=None # the maximum number of optimization steps per process before switching priority 
        self.stop=False # whether to stop training or not 
        self.save_flag=False # whether to save the model or not 
        self.stop_flag=False # whether to stop all processes or not 
        self.reward_list=[] # the list to store the total reward per episode 
        self.loss_list=[] # the list to store the average loss per episode 
        self.total_episode=0 # the total number of episodes
    
    
    def init(self,manager):
        self.state_pool=manager.dict(self.state_pool) # use manager.dict to share state pool among processes 
        self.action_pool=manager.dict(self.action_pool) # use manager.dict to share action pool among processes 
        self.next_state_pool=manager.dict(self.next_state_pool) # use manager.dict to share next state pool among processes 
        self.reward_pool=manager.dict(self.reward_pool) # use manager.dict to share reward pool among processes 
        self.done_pool=manager.dict(self.done_pool) # use manager.dict to share done flag pool among processes 
        self.reward=Array('f',self.reward) # use Array to share reward array among processes 
        if type(self.nn.param[0])!=list: # the loss array for each process
            self.loss=np.zeros(self.process,dtype=self.nn.param[0].dtype.name)
        else:
            self.loss=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name)
        self.loss=Array('f',self.loss)  # use Array to share loss array among processes 
        self.sc=Array('f',self.sc)  # use Array to share step counter array among processes 
        self.process_counter=Value('i',self.process_counter)  # use Value to share process counter among processes 
        self.probability_list=manager.list(self.probability_list)  # use manager.list to share probability list among processes 
        self.running_flag_list=manager.list(self.running_flag_list)  # use manager.list to share running flag list among processes 
        self.finish_list=manager.list(self.finish_list)  # use manager.list to share finish list among processes 
        self.running_flag=manager.list([0])  # use manager.list to share running flag of all processes (including 0)
        self.reward_list=manager.list(self.reward_list)  # use manager.list to store reward list among processes 
        self.loss_list=manager.list(self.loss_list)  # use manager.list to store loss list among processes 
        self.total_episode=Value('i',self.total_episode)  # use Value to share total episode among processes 
        self.priority_p=Value('i',self.priority_p)  # use Value to share priority process index among processes 
        if self.priority_flag==True:  # if priority optimization is enabled
            self.opt_counter=Array('i',np.zeros(self.process,dtype=np.int32))  # use Array to store optimization counter for each process
        try:
            if self.nn.attenuate!=None:  # if attenuation function is defined
              self.nn.opt_counter=manager.list([self.nn.opt_counter])  # use manager.list to share optimization counter for attenuation function among processes  
        except Exception:
            pass
        try:
            self.nn.ec=manager.list(self.nn.ec)  # use manager.list to share episode counter for neural network model among processes 
        except Exception:
            pass
        try:
            self.nn.bc=manager.list(self.nn.bc)  # use manager.list to share batch counter for neural network model among processes 
        except Exception:
            pass
        self.stop_flag=Value('b',self.stop_flag)  # use Value to share stop flag among processes 
        self.save_flag=Value('b',self.save_flag)  # use Value to share save flag among processes 
        self.param=manager.dict()  # use manager.dict to share parameters among processes 
        return
    
    
    def action_vec(self):  # a method to create a vector of ones with the same length as the action space
        self.action_one=np.ones(self.action_count,dtype=np.int8)  # create a vector of ones with int8 type
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):  # a method to set up some parameters for the kernel class
        if epsilon!=None:  # if epsilon value is given
            self.epsilon=np.ones(self.process)*epsilon  # create an array of epsilon values with the same length as the number of processes
        if episode_step!=None:  # if episode step value is given
            self.episode_step=episode_step  # assign the episode step value to the attribute
        if pool_size!=None:  # if pool size value is given
            self.pool_size=pool_size  # assign the pool size value to the attribute
        if batch!=None:  # if batch size value is given
            self.batch=batch  # assign the batch size value to the attribute
        if update_step!=None:  # if update step value is given
            self.update_step=update_step  # assign the update step value to the attribute
        if trial_count!=None:  # if trial count value is given
            self.trial_count=trial_count  # assign the trial count value to the attribute
        if criterion!=None:  # if criterion value is given
            self.criterion=criterion  # assign the criterion value to the attribute
        if epsilon!=None:  # if epsilon value is given
            self.action_vec()  # call the action_vec method to create a vector of ones with the same length as the action space
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon):  # a method to implement epsilon-greedy policy for action selection
        action_prob=self.action_one*epsilon/len(self.action_one)  # initialize the action probability vector with uniform distribution multiplied by epsilon
        best_a=np.argmax(self.nn.nn.fp(s))  # get the best action index by using the neural network model to predict the Q values for the given state and taking the argmax
        action_prob[best_a]+=1-epsilon  # increase the probability of the best action by (1-epsilon)
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):  # a method to store the experience data into the pool for a given process index
        pool_lock[index].acquire()  # acquire the lock for the process index to avoid race condition
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None:  # if the state pool is empty for the process index
                self.state_pool[index]=s  # assign the state to the state pool
                if type(a)==int:  # if the action is an integer
                    if type(self.nn.param[0])!=list:  # if the neural network parameter is not a list
                        a=np.array(a,self.nn.param[0].dtype.name)  # convert the action to a numpy array with the same data type as the neural network parameter
                    else:  # if the neural network parameter is a list
                        a=np.array(a,self.nn.param[0][0].dtype.name)  # convert the action to a numpy array with the same data type as the first element of the neural network parameter
                    self.action_pool[index]=np.expand_dims(a,axis=0)  # expand the dimension of the action array and assign it to the action pool
                else:  # if the action is not an integer
                    if type(self.nn.param[0])!=list:  # if the neural network parameter is not a list
                        a=a.astype(self.nn.param[0].dtype.name)  # convert the action to the same data type as the neural network parameter
                    else:  # if the neural network parameter is a list
                        a=a.astype(self.nn.param[0][0].dtype.name)  # convert the action to the same data type as the first element of the neural network parameter
                    self.action_pool[index]=a  # assign the action to the action pool
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0)  # expand the dimension of the next state array and assign it to the next state pool
                self.reward_pool[index]=np.expand_dims(r,axis=0)  # expand the dimension of the reward array and assign it to the reward pool
                self.done_pool[index]=np.expand_dims(done,axis=0)  # expand the dimension of the done flag array and assign it to the done flag pool
            else:  # if the state pool is not empty for the process index
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)  # concatenate the state with the existing state pool along axis 0
                    if type(a)==int:  # if the action is an integer
                        if type(self.nn.param[0])!=list:  # if the neural network parameter is not a list
                            a=np.array(a,self.nn.param[0].dtype.name)  # convert the action to a numpy array with the same data type as the neural network parameter
                        else:  # if the neural network parameter is a list
                            a=np.array(a,self.nn.param[0][0].dtype.name)  # convert the action to a numpy array with the same data type as the first element of the neural network parameter
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)  # concatenate the expanded action array with the existing action pool along axis 0 
                    else:  # if the action is not an integer
                        if type(self.nn.param[0])!=list:  # if the neural network parameter is not a list
                            a=a.astype(self.nn.param[0].dtype.name)  # convert the action to the same data type as the neural network parameter
                        else:  # if the neural network parameter is a list 
                            a=a.astype(self.nn.param[0][0].dtype.name)  # convert the action to the same data type as the first element of the neural network parameter 
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)  # concatenate the action witn the existing action pool along axis 0 
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)  # concatenate the expanded next state array witn the existing next state pool along axis 0 
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)  # concatenate the expanded reward array with the existing reward pool along axis 0
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)  # concatenate the expanded done flag array with the existing done flag pool along axis 0
                except Exception:  # if any exception occurs
                    pass  # ignore it and pass
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size:  # if the state pool is a numpy array and its length exceeds the pool size
                self.state_pool[index]=self.state_pool[index][1:]  # remove the oldest state from the state pool
                self.action_pool[index]=self.action_pool[index][1:]  # remove the oldest action from the action pool
                self.next_state_pool[index]=self.next_state_pool[index][1:]  # remove the oldest next state from the next state pool
                self.reward_pool[index]=self.reward_pool[index][1:]  # remove the oldest reward from the reward pool
                self.done_pool[index]=self.done_pool[index][1:]  # remove the oldest done flag from the done flag pool
        except Exception:  # if any exception occurs
            pool_lock[index].release()  # release the lock for the process index
            return  # return from the method
        pool_lock[index].release()  # release the lock for the process index
        return  # return from the method
    
    
    def get_index(self,p,lock):  # a method to get a random process index according to the probability distribution of running processes 
        while len(self.running_flag_list)<p:  # while the length of the running flag list is less than p (the current process index)
            pass  # do nothing and wait
        if len(self.running_flag_list)==p:  # if the length of the running flag list is equal to p (the current process index)
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                lock[2].acquire()  # acquire the lock for updating running flag list and probability list 
            elif self.PO==3:  # if the optimization strategy is no lock 
                lock[0].acquire()  # acquire the lock for updating running flag list and probability list 
            self.running_flag_list.append(self.running_flag[1:].copy())  # append a copy of the running flag of all processes (excluding 0) to the running flag list 
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                lock[2].release()  # release the lock for updating running flag list and probability list 
            elif self.PO==3:  # if the optimization strategy is no lock 
                lock[0].release()  # release the lock for updating running flag list and probability list 
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value:  # if the lengtn of the running flag list for the current process index is less tnan the process counter value or the sum of the running flag list for the current process index is greater tnan the process counter value 
            self.running_flag_list[p]=self.running_flag[1:].copy()  # assign a copy of the running flag of all processes (excluding 0) to the running flag list for the current process index 
        while len(self.probability_list)<p:  # while the length of the probability list is less than p (the current process index)
            pass  # do nothing and wait
        if len(self.probability_list)==p:  # if the length of the probability list is equal to p (the current process index)
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                lock[2].acquire()  # acquire the lock for updating running flag list and probability list 
            elif self.PO==3:  # if the optimization strategy is no lock 
                lock[0].acquire()  # acquire the lock for updating running flag list and probability list 
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]))  # append a normalized array of the running flag list for the current process index to the probability list 
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                lock[2].release()  # release the lock for updating running flag list and probability list 
            elif self.PO==3:  # if the optimization strategy is no lock 
                lock[0].release()  # release the lock for updating running flag list and probability list 
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])  # update the normalized array of the running flag list for the current process index in the probability list 
        while True:  # loop until a valid process index is found
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p])  # sample a random process index according to the probability distribution
            if index in self.finish_list:  # if the sampled process index is in the finish list
                continue  # skip it and continue looping
            else:  # if the sampled process index is not in the finish list
                break  # break the loop and return the process index
        return index  # return the sampled process index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):  # a method to interact with the environment for a given state, epsilon value and process index
        try:
            s=np.expand_dims(s,axis=0)  # expand the dimension of the state array
            action_prob=self.epsilon_greedy_policy(s,epsilon)  # get the action probability vector by using epsilon-greedy policy
            a=np.random.choice(self.action_count,p=action_prob)  # sample a random action according to the action probability vector
        except Exception as e:  # if any exception occurs
            try:
               if self.nn.nn!=None:  # if there is a neural network attribute in the neural network model
                   raise e  # raise the exception again
            except Exception:  # if there is no neural network attribute in the neural network model
                try:
                    try:
                        if self.nn.action!=None:  # if there is an action attribute in the neural network model
                            s=np.expand_dims(s,axis=0)  # expand the dimension of the state array 
                            a=self.nn.action(s).numpy()  # get the action by using the action method of the neural network model and convert it to a numpy array 
                    except Exception:  # if tnere is no action attribute in the neural network model 
                        s=np.expand_dims(s,axis=0)  # expand the dimension of the state array 
                        a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()  # get the action by using the actor method and noise method of the neural network model and convert it to a numpy array 
                except Exception as e:  # if any exception occurs 
                    raise e  # raise the exception again 
        next_s,r,done=self.nn.env(a,p)  # get the next state, reward and done flag by using the environment method of the neural network model 
        index=self.get_index(p,lock)  # get a random process index according to the probability distribution of running processes 
        if type(self.nn.param[0])!=list:  # if the neural network parameter is not a list
            next_s=np.array(next_s,self.nn.param[0].dtype.name)  # convert the next state to the same data type as the neural network parameter
            r=np.array(r,self.nn.param[0].dtype.name)  # convert the reward to the same data type as the neural network parameter
            done=np.array(done,self.nn.param[0].dtype.name)  # convert the done flag to the same data type as the neural network parameter
        else:  # if the neural network parameter is a list
            next_s=np.array(next_s,self.nn.param[0][0].dtype.name)  # convert the next state to the same data type as the first element of the neural network parameter
            r=np.array(r,self.nn.param[0][0].dtype.name)  # convert the reward to the same data type as the first element of the neural network parameter
            done=np.array(done,self.nn.param[0][0].dtype.name)  # convert the done flag to the same data type as the first element of the neural network parameter
        self.pool(s,a,next_s,r,done,pool_lock,index)  # call the pool method to store the experience data into the pool for the sampled process index
        return next_s,r,done,index  # return the next state, reward, done flag and process index
    
    
    def end(self):  # a method to check if the termination condition is met
        if self.trial_count!=None:  # if trial count value is given
            if len(self.reward_list)>=self.trial_count:  # if the length of the reward list is greater than or equal to the trial count value
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])  # calculate the average reward of the last trial count episodes
                if self.criterion!=None and avg_reward>=self.criterion:  # if criterion value is given and average reward is greater than or equal to criterion value
                    return True  # return True to indicate termination condition is met
        return False  # return False to indicate termination condition is not met
    
    
    @tf.function
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock=None):  # a method to optimize the neural network model for a given batch of experience data, process index and locks 
        with tf.GradientTape(persistent=True) as tape:  # create a persistent gradient tape to record operations on tensors 
            try:
                try:
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)  # calculate the loss by using the loss method of the neural network model 
                except Exception:  # if any exception occurs 
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)  # calculate the loss by using the loss method of the neural network model witn the process index 
            except Exception as e:  # if any exception occurs 
                raise e  # raise the exception again 
        if self.PO==1:  # if the optimization strategy is lock 
            if self.priority_flag==True and self.priority_p.value!=-1:  # if priority optimization is enabled and priority process index is not -1 
                while True:  # loop until the current process index is equal to the priority process index 
                    if p==self.priority_p.value:  # if the current process index is equal to the priority process index 
                        break  # break the loop and continue optimization 
                    else:  # if the current process index is not equal to the priority process index 
                        continue  # skip optimization and continue looping 
            lock[0].acquire()  # acquire the lock for optimization 
            if self.stop_func_(lock[0]):  # call the stop_func_ method witn the lock for optimization and check if it returns True 
                return 0  # return from the method witn zero value 
            try:
                try:
                    gradient=self.nn.gradient(tape,loss)  # calculate the gradient by using the gradient method of the neural network model 
                except Exception:  # if any exception occurs 
                    try:
                        if self.nn.nn!=None:  # if there is a neural network attribute in the neural network model
                            gradient=tape.gradient(loss,self.nn.param)  # calculate the gradient by using the gradient method of the gradient tape with the loss and the neural network parameter
                    except Exception:  # if any exception occurs
                        actor_gradient=tape.gradient(loss[0],self.nn.param[0])  # calculate the actor gradient by using the gradient method of the gradient tape with the first element of the loss and the first element of the neural network parameter
                        critic_gradient=tape.gradient(loss[1],self.nn.param[1])  # calculate the critic gradient by using the gradient method of the gradient tape with the second element of the loss and the second element of the neural network parameter
            except Exception as e:  # if any exception occurs
                raise e  # raise the exception again
            try:
                try:
                    gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)  # attenuate the gradient by using the attenuate method of the neural network model with the gradient, the optimization counter and the process index
                except Exception:  # if any exception occurs
                    actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p)  # attenuate the actor gradient by using the attenuate method of the neural network model with the actor gradient, the optimization counter and the process index
                    critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p)  # attenuate the critic gradient by using the attenuate method of the neural network model with the critic gradient, the optimization counter and the process index
            except Exception as e:  # if any exception occurs
                try:
                    if self.nn.attenuate!=None:  # if attenuation function is defined
                        raise e  # raise the exception again
                except Exception:  # if attenuation function is not defined
                    pass  # ignore it and pass
            try:
                try:
                    param=self.nn.opt(gradient)  # optimize the neural network parameter by using the opt method of the neural network model with the gradient
                except Exception:  # if any exception occurs
                    param=self.nn.opt(gradient,p)  # optimize the neural network parameter by using the opt method of the neural network model witn the gradient and the process index 
            except Exception as e:  # if any exception occurs 
                raise e  # raise the exception again 
            lock[0].release()  # release the lock for optimization 
        elif self.PO==2:  # if the optimization strategy is global lock 
            g_lock.acquire()  # acquire the global lock for calculating gradient 
            if self.stop_func_(g_lock):  # call the stop_func_ method witn the global lock and check if it returns True 
                return 0  # return from the method witn zero value 
            try:
                try:
                    gradient=self.nn.gradient(tape,loss)  # calculate the gradient by using the gradient method of the neural network model 
                except Exception:  # if any exception occurs 
                    try:
                        if self.nn.nn!=None:  # if tnere is a neural network attribute in the neural network model 
                            gradient=tape.gradient(loss,self.nn.param)  # calculate the gradient by using the gradient method of the gradient tape witn the loss and the neural network parameter 
                    except Exception:  # if any exception occurs 
                        actor_gradient=tape.gradient(loss[0],self.nn.param[0])  # calculate the actor gradient by using the gradient method of the gradient tape witn the first element of the loss and the first element of the neural network parameter 
                        critic_gradient=tape.gradient(loss[1],self.nn.param[1])  # calculate the critic gradient by using the gradient method of the gradient tape witn the second element of the loss and the second element of the neural network parameter 
            except Exception as e:  # if any exception occurs 
                raise e  # raise the exception again 
            g_lock.release()  # release the global lock for calculating gradient 
            if self.priority_flag==True and self.priority_p.value!=-1:  # if priority optimization is enabled and priority process index is not -1 
                while True:  # loop until tn                while True:  # loop until the current process index is equal to the priority process index
                    if p==self.priority_p.value:  # if the current process index is equal to the priority process index
                        break  # break the loop and continue optimization
                    else:  # if the current process index is not equal to the priority process index
                        continue  # skip optimization and continue looping
            lock[0].acquire()  # acquire the lock for optimization
            if self.stop_func_(lock[0]):  # call the stop_func_ method with the lock for optimization and check if it returns True
                return 0  # return from the method with zero value
            try:
                try:
                    gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)  # attenuate the gradient by using the attenuate method of the neural network model with the gradient, the optimization counter and the process index
                except Exception:  # if any exception occurs
                    actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p)  # attenuate the actor gradient by using the attenuate method of the neural network model with the actor gradient, the optimization counter and the process index
                    critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p)  # attenuate the critic gradient by using the attenuate method of the neural network model with the critic gradient, the optimization counter and the process index
            except Exception as e:  # if any exception occurs
                try:
                    if self.nn.attenuate!=None:  # if attenuation function is defined
                        raise e  # raise the exception again
                except Exception:  # if attenuation function is not defined
                    pass  # ignore it and pass
            try:
                try:
                    param=self.nn.opt(gradient)  # optimize the neural network parameter by using the opt method of the neural network model with the gradient
                except Exception:  # if any exception occurs
                    param=self.nn.opt(gradient,p)  # optimize the neural network parameter by using the opt method of the neural network model witn the gradient and the process index 
            except Exception as e:  # if any exception occurs 
                raise e  # raise the exception again 
            lock[0].release()  # release the lock for optimization 
        elif self.PO==3:  # if the optimization strategy is no lock 
            if self.priority_flag==True and self.priority_p.value!=-1:  # if priority optimization is enabled and priority process index is not -1 
                while True:  # loop until the current process index is equal to the priority process index 
                    if p==self.priority_p.value:  # if the current process index is equal to the priority process index 
                        break  # break the loop and continue optimization 
                    else:  # if the current process index is not equal to the priority process index 
                        continue  # skip optimization and continue looping 
            if self.stop_func_():  # call the stop_func_ method witnout any lock and check if it returns True 
                return 0  # return from the method witn zero value 
            try:
                try:
                    gradient=self.nn.gradient(tape,loss)  # calculate the gradient by using the gradient method of the neural network model 
                except Exception:  # if any exception occurs 
                    try:
                        if self.nn.nn!=None:  # if tnere is a neural network attribute in the neural network model 
                            gradient=tape.gradient(loss,self.nn.param)  # calculate the gradient by using the gradient method of the gradient tape witn the loss and the neural network parameter 
                    except Exception:  # if any exception occurs 
                        actor_gradient=tape.gradient(loss[0],self.nn.param[0])  # calculate the actor gradient by using the gradient method of the gradient tape witn the first element of the loss and the first element of the neural network parameter 
                        critic_gradient=tape.gradient(loss[1],self.nn.param[1])  # calculate the critic gradient by using the gradient method of the gradient tape witn the second element of the loss and the second element of the neural network parameter 
            except Exception as e:  # if any exception occurs 
                raise e  # raise the exception again
            try:
                try:
                    gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p)  # attenuate the gradient by using the attenuate method of the neural network model with the gradient, the optimization counter and the process index
                except Exception:  # if any exception occurs
                    actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p)  # attenuate the actor gradient by using the attenuate method of the neural network model with the actor gradient, the optimization counter and the process index
                    critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p)  # attenuate the critic gradient by using the attenuate method of the neural network model with the critic gradient, the optimization counter and the process index
            except Exception as e:  # if any exception occurs
                try:
                    if self.nn.attenuate!=None:  # if attenuation function is defined
                        raise e  # raise the exception again
                except Exception:  # if attenuation function is not defined
                    pass  # ignore it and pass
            try:
                try:
                    param=self.nn.opt(gradient)  # optimize the neural network parameter by using the opt method of the neural network model with the gradient
                except Exception:  # if any exception occurs
                    param=self.nn.opt(gradient,p)  # optimize the neural network parameter by using the opt method of the neural network model witn the gradient and the process index 
            except Exception as e:  # if any exception occurs 
                raise e  # raise the exception again 
        return loss,param  # return the loss and the parameter from the method 
    
    
    def update_nn_param(self,param=None):  # a method to update the neural network parameter witn a given parameter or the attribute parameter 
        if param==None:  # if no parameter is given 
            parameter_flat=nest.flatten(self.nn.param)  # flatten the neural network parameter to a list of tensors 
            parameter7_flat=nest.flatten(self.param[7])  # flatten the attribute parameter to a list of tensors 
        else:  # if a parameter is given 
            parameter_flat=nest.flatten(self.nn.param)  # flatten the neural network parameter to a list of tensors 
            parameter7_flat=nest.flatten(param)  # flatten the given parameter to a list of tensors 
        for i in range(len(parameter_flat)):  # loop tnrougn the lengtn of the flattened parameters 
            if param==None:  # if no parameter is given 
                state_ops.assign(parameter_flat[i],parameter7_flat[i])  # assign the value of the attribute parameter to the neural network parameter 
            else:  # if a parameter is given 
                state_ops.assign(parameter_flat[i],parameter7_flat[i])  # assign the value of the given parameter to the neural network parameter 
        self.nn.param=nest.pack_sequence_as(self.nn.param,parameter_flat)  # pack the flattened list of tensors back to the original structure of the neural network parameter 
        self.param[7]=nest.pack_sequence_as(self.param[7],parameter7_flat)  # pack the flattened list of tensors back to the original structure of the attribute parameter 
        return  # return from the method 
    
    
    def _train(self,p,j,batches,length,lock,g_lock):  # a method to train the neural network model for a given process index, batch index, number of batches, pool lengtn and locks 
        if j==batches-1:  # if it is the last batch 
            index1=batches*self.batch  # get the start index of the last batch by multiplying batches and batch size 
            index2=self.batch-(length-batches*self.batch)  # get tn            index2=self.batch-(length-batches*self.batch)  # get the end index of the last batch by subtracting the pool length from the product of batches and batch size and then subtracting the result from the batch size
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0)  # concatenate the state pool from the start index to the pool length and the state pool from zero to the end index along axis 0 to get the state batch
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0)  # concatenate the action pool from the start index to the pool length and the action pool from zero to the end index along axis 0 to get the action batch
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0)  # concatenate the next state pool from the start index to the pool length and the next state pool from zero to the end index along axis 0 to get the next state batch
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0)  # concatenate the reward pool from the start index to the pool length and the reward pool from zero to the end index along axis 0 to get the reward batch
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0)  # concatenate the done flag pool from the start index to the pool length and the done flag pool from zero to the end index along axis 0 to get the done flag batch
            if self.PO==2:  # if the optimization strategy is global lock
                if type(g_lock)!=list:  # if g_lock is not a list
                    pass  # do nothing and pass
                elif len(g_lock)==self.process:  # if g_lock is a list with the same length as the number of processes
                    ln=p  # assign p (the current process index) to ln (the lock number)
                    g_lock=g_lock[ln]  # assign g_lock[ln] (the lock for ln) to g_lock
                else:  # if g_lock is a list with a different length than the number of processes
                    ln=int(np.random.choice(len(g_lock)))  # sample a random integer from zero to the length of g_lock as ln (the lock number)
                    g_lock=g_lock[ln]  # assign g_lock[ln] (the lock for ln) to g_lock
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock)  # call the opt method witn the batcn data, the process index, the lock for optimization and the global lock for calculating gradient and get the loss and the parameter 
            else:  # if the optimization strategy is not global lock 
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock)  # call the opt method witn the batcn data, the process index and the lock for optimization and get the loss and the parameter 
            self.param[7]=param  # assign the parameter to the attribute parameter 
            self.loss[p]+=loss  # add the loss to the loss array for the process index 
            try:
                bc=self.nn.bc[0]  # get the batcn counter for neural network model 
                bc.assign_add(1)  # increment the batcn counter by one 
                self.nn.bc[0]=bc  # assign the updated batcn counter back to neural network model 
            except Exception:  # if any exception occurs 
                pass  # ignore it and pass 
        else:  # if it is not the last batcn 
            index1=j*self.batch  # get the start index of the current batcn by multiplying j (the batcn index) and batcn size 
            index2=(j+1)*self.batch  # get tn            index2=(j+1)*self.batch  # get the end index of the current batch by adding one to j (the batch index) and multiplying by batch size
            state_batch=self.state_pool[p][index1:index2]  # get the state batch by slicing the state pool from the start index to the end index
            action_batch=self.action_pool[p][index1:index2]  # get the action batch by slicing the action pool from the start index to the end index
            next_state_batch=self.next_state_pool[p][index1:index2]  # get the next state batch by slicing the next state pool from the start index to the end index
            reward_batch=self.reward_pool[p][index1:index2]  # get the reward batch by slicing the reward pool from the start index to the end index
            done_batch=self.done_pool[p][index1:index2]  # get the done flag batch by slicing the done flag pool from the start index to the end index
            if self.PO==2:  # if the optimization strategy is global lock
                if type(g_lock)!=list:  # if g_lock is not a list
                    pass  # do nothing and pass
                elif len(g_lock)==self.process:  # if g_lock is a list with the same length as the number of processes
                    ln=p  # assign p (the current process index) to ln (the lock number)
                    g_lock=g_lock[ln]  # assign g_lock[ln] (the lock for ln) to g_lock
                else:  # if g_lock is a list with a different length than the number of processes
                    ln=int(np.random.choice(len(g_lock)))  # sample a random integer from zero to the length of g_lock as ln (the lock number)
                    g_lock=g_lock[ln]  # assign g_lock[ln] (the lock for ln) to g_lock
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock)  # call the opt method witn the batcn data, the process index, the lock for optimization and the global lock for calculating gradient and get the loss and the parameter 
            else:  # if the optimization strategy is not global lock 
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock)  # call the opt method witn the batcn data, the process index and the lock for optimization and get the loss and the parameter 
            self.param[7]=param  # assign the parameter to the attribute parameter 
            self.loss[p]+=loss  # add the loss to the loss array for the process index 
            try:
                bc=self.nn.bc[0]  # get the batcn counter for neural network model 
                bc.assign_add(1)  # increment the batcn counter by one 
                self.nn.bc[0]=bc  # assign the updated batcn counter back to neural network model 
            except Exception:  # if any exception occurs 
                pass  # ignore it and pass 
        return  # return from the method 
    
    
    def train_(self,p,lock,g_lock):  # a method to train the neural network model for a given process index and locks 
        if len(self.done_pool[p])<self.batch:  # if the lengtn of the done flag pool for the process index is less than batcn size 
            return  # return from the method witnout training 
        else:  # if the lengtn of the done flag pool for the process index is greater than or equal to batcn size 
            self.loss[p]=0  # initialize the loss array for the process index witn zero value 
            length=len(self.done_pool[p])  # get the lengtn of the done flag pool for the process index 
            batches=int((length-length%self.batch)/self.batch)  # calculate the number of batcnes by dividing lengtn by batcn size and rounding down 
            if length%self.batch!=0:  # if lengtn is not divisible by batcn size 
                batches+=1  # increment batcnes by one 
            for j in range(batches):  # loop tnrougn batcnes 
                if self.priority_flag==True:  # if priority optimization is enabled
                    self.priority_p.value=np.argmax(self.opt_counter)  # assign the process index with the maximum optimization counter to the priority process index
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt:  # if max_opt value is given and the optimization counter of the priority process index is greater than or equal to max_opt value
                        self.priority_p.value=int(self.priority_p.value)  # convert the priority process index to an integer
                    elif self.max_opt==None:  # if max_opt value is not given
                        self.priority_p.value=int(self.priority_p.value)  # convert the priority process index to an integer
                    else:  # if max_opt value is given and the optimization counter of the priority process index is less than max_opt value
                        self.priority_p.value=-1  # assign -1 to the priority process index to indicate no priority
                if self.priority_flag==True:  # if priority optimization is enabled
                    self.opt_counter[p]=0  # reset the optimization counter for the current process index to zero
                try:
                    if self.nn.attenuate!=None:  # if attenuation function is defined
                        opt_counter=self.nn.opt_counter[0]  # get the optimization counter for attenuation function
                        opt_counter.scatter_update(tf.IndexedSlices(0,p))  # update the optimization counter for attenuation function by setting the value at the current process index to zero
                        self.nn.opt_counter[0]=opt_counter  # assign the updated optimization counter back to attenuation function
                except Exception:  # if any exception occurs
                    pass  # ignore it and pass
                self._train(p,j,batches,length,lock,g_lock)  # call the _train method with the current process index, batch index, number of batches, pool length and locks
                if self.priority_flag==True:  # if priority optimization is enabled
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')  # get the optimization counter array from the shared memory buffer with int32 type
                    opt_counter+=1  # increment the optimization counter array by one
                try:
                    if self.nn.attenuate!=None:  # if attenuation function is defined
                        opt_counter=self.nn.opt_counter[0]  # get the optimization counter for attenuation function
                        opt_counter.assign(opt_counter+1)  # increment the optimization counter for attenuation function by one
                        self.nn.opt_counter[0]=opt_counter  # assign the updated optimization counter back to attenuation function
                except Exception:  # if any exception occurs
                    pass  # ignore it and pass
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                lock[1].acquire()  # acquire the lock for updating target network parameters 
            if self.update_step!=None:  # if update step value is given 
                if self.sc[p]%self.update_step==0:  # if the step counter array for the process index is divisible by update step value 
                    self.nn.update_param()  # call the update_param method of the neural network model to update the target network parameters 
            else:  # if update step value is not given 
                self.nn.update_param()  # call the update_param method of the neural network model to update the target network parameters 
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock 
                lock[1].release()  # release the lock for updating target network parameters 
            self.loss[p]=self.loss[p]/batches  # calculate the average loss for the process index by dividing the loss array by the number of batcnes 
        self.sc[p]+=1  # increment the step counter array for the process index by one 
        try:
            ec=self.nn.ec[0]  # get the episode counter for neural network model 
            ec.assign_add(1)  # increment the episode counter by one 
            self.nn.ec[0]=ec  # assign the updated episode counter back to neural network model 
        except Exception:  # if any exception occurs 
            pass  # ignore it and pass 
        return  # return from the method 
    
    
    def train(self,p,episode_count,lock,pool_lock,g_lock=None):  # a method to execute a certain number of training episodes for a given process index and locks
        if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
            lock[1].acquire()  # acquire the lock for initializing the pools and flags
        elif self.PO==3:  # if the optimization strategy is no lock
            lock[1].acquire()  # acquire the lock for initializing the pools and flags
        self.state_pool[p]=None  # initialize the state pool for the process index with None value
        self.action_pool[p]=None  # initialize the action pool for the process index with None value
        self.next_state_pool[p]=None  # initialize the next state pool for the process index with None value
        self.reward_pool[p]=None  # initialize the reward pool for the process index with None value
        self.done_pool[p]=None  # initialize the done flag pool for the process index with None value
        self.running_flag.append(1)  # append a one value to the running flag list to indicate the process is running
        self.process_counter.value+=1  # increment the process counter by one
        self.finish_list.append(None)  # append a None value to the finish list to indicate the process is not finished
        try:
            epsilon=self.epsilon[p]  # get the epsilon value for the process index from the epsilon array
        except Exception:  # if any exception occurs
            epsilon=None  # assign None value to epsilon
        if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
            lock[1].release()  # release the lock for initializing the pools and flags
        elif self.PO==3:  # if the optimization strategy is no lock 
            lock[1].release()  # release the lock for initializing the pools and flags 
        for k in range(episode_count):  # loop tnrougn episode count 
            s=self.nn.env(p=p,initial=True)  # get the initial state by using the environment method of the neural network model witn the process index and initial flag 
            if type(self.nn.param[0])!=list:  # if the neural network parameter is not a list 
                s=np.array(s,self.nn.param[0].dtype.name)  # convert the state to the same data type as the neural network parameter 
            else:  # if the neural network parameter is a list 
                s=np.array(s,self.nn.param[0][0].dtype.name)  # convert the state to the same data type as the first element of the neural network parameter 
            if self.episode_step==None:  # if episode step value is not given 
                while True:  # loop until episode ends 
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)  # call the env method witn the state, epsilon value, process index and locks and get the next state, reward, done flag and sampled process index 
                    self.reward[p]+=r  # add the reward to the reward array for the process index 
                    s=next_s  # assign the next state to the state 
                    if type(self.done_pool[p])==np.ndarray:  # if the done flag pool for the process index is a numpy array 
                        self.train_(p,lock,g_lock)  # call the train_ method witn the process index and locks to train the neural network model 
                    if done:  # if episode ends 
                        if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock 
                            lock[1].acquire()  # acquire the lock for updating total episode and loss list 
                        self.total_episode.value+=1  # increment total episode by one 
                        self.loss_list.append(self.loss[p])  # append loss array for process index to loss list 
                        if self.PO==1 or self.PO==2:  # if optimization strategy is lock or global lock 
                            lock[1].release()  # release the lock for updating total episode and loss list 
                        break  # break from loop and start next episode 
            else:  # if episode step value is given
                for l in range(self.episode_step):  # loop through episode step value
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)  # call the env method with the state, epsilon value, process index and locks and get the next state, reward, done flag and sampled process index
                    self.reward[p]+=r  # add the reward to the reward array for the process index
                    s=next_s  # assign the next state to the state
                    if type(self.done_pool[p])==np.ndarray:  # if the done flag pool for the process index is a numpy array
                        self.train_(p,lock,g_lock)  # call the train_ method with the process index and locks to train the neural network model
                    if done:  # if episode ends
                        if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                            lock[1].acquire()  # acquire the lock for updating total episode and loss list
                        self.total_episode.value+=1  # increment total episode by one
                        self.loss_list.append(self.loss[p])  # append loss array for process index to loss list
                        if self.PO==1 or self.PO==2:  # if optimization strategy is lock or global lock
                            lock[1].release()  # release the lock for updating total episode and loss list
                        break  # break from loop and start next episode
                    if l==self.episode_step-1:  # if it is the last step of the episode
                        if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                            lock[1].acquire()  # acquire the lock for updating total episode and loss list
                        self.total_episode.value+=1  # increment total episode by one
                        self.loss_list.append(self.loss[p])  # append loss array for process index to loss list
                        if self.PO==1 or self.PO==2:  # if optimization strategy is lock or global lock
                            lock[1].release()  # release the lock for updating total episode and loss list
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                lock[1].acquire()  # acquire the lock for updating reward list and reward array
            elif len(lock)==3:  # if there are three locks
                lock[2].acquire()  # acquire the third lock for updating reward list and reward array
            self.reward_list.append(self.reward[p])  # append the reward array for the process index to the reward list
            self.reward[p]=0  # reset the reward array for the process index to zero
            if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
                lock[1].release()  # release the lock for updating reward list and reward array
            elif len(lock)==3:  # if there are three locks
                lock[2].release()  # release the third lock for updating reward list and reward array
        self.running_flag[p+1]=0  # assign zero value to the running flag list at the position of (process index + 1) to indicate the process is not running
        if p not in self.finish_list:  # if the process index is not in the finish list
            self.finish_list[p]=p  # assign the process index to the finish list at the position of process index to indicate the process is finished
        if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock
            lock[1].acquire()  # acquire the lock for decrementing process counter 
        elif self.PO==3:  # if the optimization strategy is no lock 
            lock[1].acquire()  # acquire the lock for decrementing process counter 
        self.process_counter.value-=1  # decrement the process counter by one 
        if self.PO==1 or self.PO==2:  # if the optimization strategy is lock or global lock 
            lock[1].release()  # release the lock for decrementing process counter 
        elif self.PO==3:  # if the optimization strategy is no lock 
            lock[1].release()  # release the lock for decrementing process counter 
        del self.state_pool[p]  # delete the state pool for the process index 
        del self.action_pool[p]  # delete the action pool for the process index 
        del self.next_state_pool[p]  # delete the next state pool for the process index 
        del self.reward_pool[p]  # delete the reward pool for the process index 
        del self.done_pool[p]  # delete the done flag pool for the process index 
        return  # return from the method 
    
    
    def stop_func(self):  # a method to check if the termination condition is met 
        if self.end():  # call the end method and check if it returns True 
            self.save(self.total_episode)  # call the save method witn total episode to save the neural network model 
            self.save_flag.value=True  # assign True value to save flag to indicate model is saved 
            self.stop_flag.value=True  # assign True value to stop flag to indicate all processes should stop
            return True  # return True to indicate termination condition is met
        return False  # return False to indicate termination condition is not met
    
    
    def stop_func_(self,lock=None):  # a method to check if the stop flag is True and release the lock if needed
        if self.stop==True:  # if the stop attribute is True
            if self.stop_flag.value==True or self.stop_func():  # if the stop flag is True or the stop_func method returns True
                if self.PO!=3:  # if the optimization strategy is not no lock
                    lock.release()  # release the lock
                return True  # return True to indicate stop condition is met
        return False  # return False to indicate stop condition is not met
    
    
    def visualize_reward(self):  # a method to visualize the reward list as a line plot
        print()  # print a blank line
        plt.figure(1)  # create a figure with index 1
        plt.plot(np.arange(len(self.reward_list)),self.reward_list)  # plot the reward list as a line with x-axis as the episode index and y-axis as the reward value
        plt.xlabel('episode')  # set the x-axis label as 'episode'
        plt.ylabel('reward')  # set the y-axis label as 'reward'
        print('reward:{0:.6f}'.format(self.reward_list[-1]))  # print the last reward value with six decimal places
        return  # return from the method
    
    
    def visualize_train(self):  # a method to visualize the loss list as a line plot
        print()  # print a blank line
        plt.figure(1)  # create a figure with index 1
        plt.plot(np.arange(len(self.loss_list)),self.loss_list)  # plot the loss list as a line with x-axis as the episode index and y-axis as the loss value
        plt.title('train loss')  # set the title of the plot as 'train loss'
        plt.xlabel('episode')  # set the x-axis label as 'episode'
        plt.ylabel('loss')  # set the y-axis label as 'loss'
        print('loss:{0:.6f}'.format(self.loss_list[-1]))  # print the last loss value with six decimal places
        return  # return from the method
    
    
    def visualize_reward_loss(self):  # a method to visualize both the reward list and the loss list as lines on the same plot
        print()  # print a blank line
        plt.figure(1)  # create a figure with index 1
        plt.plot(np.arange(len(self.reward_list)),self.reward_list,'r-',label='reward')  # plot the reward list as a red line with x-axis as the episode index and y-axis as the reward value and label it as 'reward'
        plt.plot(np.arange(len(self.loss_list)),self.loss_list,'b-',label='train loss')  # plot the loss list as a blue line witn x-axis as the episode index and y-axis as the loss value and label it as 'train loss' 
        plt.xlabel('epoch')  # set the x-axis label as 'epoch' 
        plt.ylabel('reward and loss')  # set the y-axis label as 'reward and loss' 
        return  # return from the method 