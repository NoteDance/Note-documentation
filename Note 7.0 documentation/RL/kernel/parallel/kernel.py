# This is a code snippet that defines a kernel class for multi-process reinforcement learning
# using TensorFlow and multiprocessing

import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Value,Array
import numpy as np
import statistics


class kernel:
    def __init__(self,nn=None,process=None):
        self.nn=nn # the neural network model
        if process!=None:
            self.reward=np.zeros(process,dtype=np.float32) # the reward array for each process
            self.sc=np.zeros(process,dtype=np.int32) # the step counter array for each process
        self.epsilon=None # the exploration rate array for each process
        self.episode_step=None # the maximum number of steps per episode
        self.pool_size=None # the maximum size of the pool
        self.batch=None # the batch size for training
        self.update_step=None # the update frequency for the target network
        self.trial_count=None # the number of trials to compute the average reward
        self.process=process # the number of processes
        self.PO=None # the parallel optimization mode (1, 2, or 3)
        self.priority_flag=False # the priority flag for optimization order
        self.max_opt=None # the maximum number of optimization steps per process per episode
        self.stop=False # the stop flag for training
        self.s=None # the state array for online training mode
        self.filename='save.dat' # the file name to save parameters to file
    
    
    def init(self,manager):
        self.state_pool=manager.dict({}) # create a shared memory space for state pool using manager object 
        self.action_pool=manager.dict({}) # create a shared memory space for action pool using manager object 
        self.next_state_pool=manager.dict({}) # create a shared memory space for next state pool using manager object 
        self.reward_pool=manager.dict({}) # create a shared memory space for reward pool using manager object 
        self.done_pool=manager.dict({}) # create a shared memory space for done pool using manager object 
        self.reward=Array('f',self.reward) # create a shared memory space for reward array using Array object 
        if type(self.nn.param[0])!=list:
            self.loss=np.zeros(self.process,dtype=self.nn.param[0].dtype.name) # create a loss array with the same data type as neural network parameters 
        else:
            self.loss=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name) # create a loss array with the same data type as neural network parameters 
        self.loss=Array('f',self.loss)
        self.sc=Array('i',self.sc) # create a shared memory space for step counter array using Array object 
        self.process_counter=Value('i',0) # create a shared memory space for process counter using Value object 
        self.probability_list=manager.list([]) # create a shared memory space for probability list using manager object 
        self.running_flag_list=manager.list([]) # create a shared memory space for running flag list using manager object 
        self.finish_list=manager.list([]) # create a shared memory space for finish flag list using manager object 
        self.running_flag=manager.list([0]) # create a shared memory space for running flag array using manager object 
        self.reward_list=manager.list([]) # create a shared memory space for reward list using manager object 
        self.loss_list=manager.list([]) # create a shared memory space for loss list using manager object 
        self.total_episode=Value('i',0) # create a shared memory space for total episode counter using Value object 
        self.priority_p=Value('i',0) # create a shared memory space for priority index using Value object 
        if self.priority_flag==True:
            self.opt_counter=Array('i',np.zeros(self.process,dtype=np.int32)) # create a shared memory space for optimization counter array using Array object 
        try:
            self.nn.opt_counter=manager.list([self.nn.opt_counter])  # create a shared memory space for neural network optimization counter using manager object 
        except Exception:
            self.opt_counter_=manager.list() # create an empty list to store the exception
        try:
            self.nn.ec=manager.list([self.nn.ec])  # create a shared memory space for neural network episode counter using manager object 
        except Exception:
            self.ec_=manager.list() # create an empty list to store the exception
        try:
            self.nn.bc=manager.list([self.nn.bc]) # create a shared memory space for neural network batch counter using manager object 
        except Exception:
            self.bc_=manager.list() # create an empty list to store the exception
        self.episode_=Value('i',self.total_episode) # create a shared memory space for episode counter using Value object 
        self.stop_flag=Value('b',False) # create a shared memory space for stop flag using Value object 
        self.save_flag=Value('b',False) # create a shared memory space for save flag using Value object 
        self.file_list=manager.list([]) # create an empty list to store the file names
        self.param=manager.dict() # create an empty dictionary to store the parameters
        return
    
    
    def init_online(self,manager):
        self.nn.train_loss_list=manager.list([]) # create an empty list to store the training losses
        self.nn.counter=manager.list([]) # create an empty list to store the training counters
        self.nn.exception_list=manager.list([]) # create an empty list to store the exceptions
        self.param=manager.dict() # create an empty dictionary to store the parameters
        self.param[7]=self.nn.param # assign the neural network parameters to the dictionary
        return
    
    
    def action_vec(self):
        self.action_one=np.ones(self.action_count,dtype=np.int8) # create an action vector with ones
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
        if epsilon!=None:
            self.epsilon=np.ones(self.process)*epsilon # assign the exploration rate array with epsilon
        if episode_step!=None:
            self.episode_step=episode_step # assign the maximum number of steps per episode
        if pool_size!=None:
            self.pool_size=pool_size # assign the maximum size of the pool
        if batch!=None:
            self.batch=batch # assign the batch size for training
        if update_step!=None:
            self.update_step=update_step # assign the update frequency for the target network
        if trial_count!=None:
            self.trial_count=trial_count # assign the number of trials to compute the average reward
        if criterion!=None:
            self.criterion=criterion # assign the criterion to judge if the agent solves the environment
        if epsilon!=None:
            self.action_vec() # create an action vector with ones
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon):
        action_prob=self.action_one*epsilon/len(self.action_one) # create a uniform action probability vector based on epsilon
        best_a=np.argmax(self.nn.nn.fp(s)) # get the best action based on the neural network output
        action_prob[best_a]+=1-epsilon # increase the probability of the best action by 1-epsilon
        return action_prob # return the action probability vector
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        pool_lock[index].acquire() # acquire the lock for the pool of the current process
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None: # check if the pool is empty
                self.state_pool[index]=s # assign the state to the pool
                if type(a)==int: # check if the action is an integer
                    a=np.array(a) # convert the action to an array
                    self.action_pool[index]=np.expand_dims(a,axis=0) # expand the dimension of the action and assign it to the pool
                else: # otherwise, assume the action is an array
                    self.action_pool[index]=a # assign the action to the pool
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0) # expand the dimension of the next state and assign it to the pool
                self.reward_pool[index]=np.expand_dims(r,axis=0) # expand the dimension of the reward and assign it to the pool
                self.done_pool[index]=np.expand_dims(done,axis=0) # expand the dimension of the done flag and assign it to the pool
            else: # otherwise, assume the pool is not empty
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0) # concatenate the state with the existing pool
                    if type(a)==int: # check if the action is an integer
                        a=np.array(a) # convert the action to an array
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0) # expand and concatenate the action with the existing pool
                    else: # otherwise, assume the action is an array
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0) # concatenate the action with the existing pool
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0) # expand and concatenate the next state with the existing pool
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0) # expand and concatenate the reward with the existing pool
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0) # expand and concatenate the done flag with the existing pool
                except Exception:
                    pass # ignore any exception
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size: # check if the pool size exceeds the maximum size
                self.state_pool[index]=self.state_pool[index][1:] # delete the oldest state from the pool
                self.action_pool[index]=self.action_pool[index][1:] # delete the oldest action from the pool
                self.next_state_pool[index]=self.next_state_pool[index][1:] # delete the oldest next state from the pool
                self.reward_pool[index]=self.reward_pool[index][1:] # delete the oldest reward from the pool
                self.done_pool[index]=self.done_pool[index][1:] # delete the oldest done flag from the pool
        except Exception:
            pool_lock[index].release() # release the lock for the pool of the current process
            return
        pool_lock[index].release() # release the lock for the pool of the current process
        return
    
    
    def get_index(self,p,lock):
        while len(self.running_flag_list)<p: # wait until enough processes are running
            pass
        if len(self.running_flag_list)==p: # check if this is the first time to get index for this process
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[2].acquire() # acquire the lock for running flag list and probability list
            elif self.PO==3: # check if parallel optimization mode is 3
                lock[0].acquire() # acquire the lock for running flag list and probability list
            self.running_flag_list.append(self.running_flag[1:].copy()) # append a copy of running flag array to running flag list
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[2].release() # release the lock for running flag list and probability list
            elif self.PO==3: # check if parallel optimization mode is 3
                lock[0].release() # release the lock for running flag list and probability list
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value: # check if running flag list needs to be updated
            self.running_flag_list[p]=self.running_flag[1:].copy() # update running flag list with a copy of running flag array
        while len(self.probability_list)<p: # wait until enough processes are running
            pass
        if len(self.probability_list)==p: # check if this is the first time to get index for this process
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[2].acquire() # acquire the lock for running flag list and probability list
            elif self.PO==3: # check if parallel optimization mode is 3
                lock[0].acquire() # acquire the lock for running flag list and probability list
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])) # append a probability vector based on running flag list to probability list 
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[2].release() # release the lock for running flag list and probability list
            elif self.PO==3: # check if parallel optimization mode is 3
                lock[0].release() # release the lock for running flag list and probability list
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]) # update probability vector based on running flag list 
        while True:
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p]) # sample an index based on probability vector 
            if index in self.finish_list: # check if the sampled index is already finished 
                continue # try again 
            else:
                break # break the loop 
        return index # return the sampled index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):
        if hasattr(self.nn,'nn'): # check if neural network model has a nn attribute (single network)
            s=np.expand_dims(s,axis=0) # expand the dimension of state 
            action_prob=self.epsilon_greedy_policy(s,epsilon) # get action probability vector using epsilon-greedy policy 
            a=np.random.choice(self.action_count,p=action_prob)
        else: # otherwise, assume neural network model does not have a nn attribute (actor-critic network)
            if hasattr(self.nn,'action'): # check if neural network model has an action attribute (deterministic policy)
                s=np.expand_dims(s,axis=0) # expand the dimension of state 
                a=self.nn.action(s).numpy() # get action using neural network model
            else: # otherwise, assume neural network model does not have an action attribute (stochastic policy)
                s=np.expand_dims(s,axis=0) # expand the dimension of state 
                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy() # get action using neural network model and noise
        next_s,r,done=self.nn.env(a,p) # execute the action and get the next state, reward, and done flag
        index=self.get_index(p,lock) # get an index for the pool of the current process
        if type(self.nn.param[0])!=list:
            next_s=np.array(next_s,self.nn.param[0].dtype.name) # convert the next state to an array with the same data type as neural network parameters 
            r=np.array(r,self.nn.param[0].dtype.name) # convert the reward to an array with the same data type as neural network parameters 
            done=np.array(done,self.nn.param[0].dtype.name) # convert the done flag to an array with the same data type as neural network parameters 
        else:
            next_s=np.array(next_s,self.nn.param[0][0].dtype.name) # convert the next state to an array with the same data type as neural network parameters 
            r=np.array(r,self.nn.param[0][0].dtype.name) # convert the reward to an array with the same data type as neural network parameters 
            done=np.array(done,self.nn.param[0][0].dtype.name) # convert the done flag to an array with the same data type as neural network parameters 
        self.pool(s,a,next_s,r,done,pool_lock,index) # store the transition to the pool of the current process
        return next_s,r,done,index # return the next state, reward, done flag, and index
    
    
    def end(self):
        if self.trial_count!=None: # check if trial count is set
            if len(self.reward_list)>=self.trial_count: # check if enough rewards are available
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:]) # compute the average reward of the last trial count episodes
                if self.criterion!=None and avg_reward>=self.criterion: # check if criterion is set and average reward is greater than or equal to criterion
                    return True # return True to indicate that the agent solves the environment
        return False # return False to indicate that the agent does not solve the environment
    
    
    @tf.function(jit_compile=True)
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock=None):
        with tf.GradientTape(persistent=True) as tape: # create a gradient tape object to record gradients
            try:
                try:
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # compute the loss using neural network model
                except Exception:
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p) # compute the loss using neural network model and process index
            except Exception as e:
                raise e # raise any exception
        if self.PO==1: # check if parallel optimization mode is 1
            if self.priority_flag==True and self.priority_p.value!=-1: # check if priority flag is True and priority index is not -1
                while True:
                    if p==self.priority_p.value: # check if current process index is equal to priority index
                        break # break the loop 
                    else:
                        continue # try again 
            lock[0].acquire() # acquire the lock for optimization operation
            if self.stop_func_(lock[0]): # check if stop flag is True
                return 0 # return 0 to indicate that optimization operation is stopped
            if hasattr(self.nn,'gradient'): # check if neural network model has a gradient attribute (custom gradient function)
                gradient=self.nn.gradient(tape,loss) # compute the gradient using neural network model and loss
            else: 
                if hasattr(self.nn,'nn'): # check if neural network model has a nn attribute (single network)
                    gradient=tape.gradient(loss,self.nn.param) # compute the gradient using loss and neural network parameters
                else: # otherwise, assume neural network model does not have a nn attribute (actor-critic network)
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0]) # compute the actor gradient using loss and actor parameters
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1]) # compute the critic gradient using loss and critic parameters
            try:
                if hasattr(self.nn,'attenuate'): # check if neural network model has an attenuate attribute (custom attenuate function)
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # attenuate the gradient using neural network model, optimization counter, and process index
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p) # attenuate the actor gradient using neural network model, optimization counter, and process index
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p) # attenuate the critic gradient using neural network model, optimization counter, and process index
            except Exception as e:
                raise e # raise any exception
            try:
                try:
                    param=self.nn.opt(gradient) # update the neural network parameters using optimizer and gradient
                except Exception:
                    param=self.nn.opt(gradient,p) # update the neural network parameters using optimizer, gradient, and process index
            except Exception as e:
                raise e # raise any exception
            lock[0].release() # release the lock for optimization operation
        elif self.PO==2: # check if parallel optimization mode is 2
            g_lock.acquire() # acquire the global lock for gradient computation
            if self.stop_func_(g_lock): # check if stop flag is True
                return 0 # return 0 to indicate that optimization operation is stopped
            if hasattr(self.nn,'gradient'): # check if neural network model has a gradient attribute (custom gradient function)
                gradient=self.nn.gradient(tape,loss) # compute the gradient using neural network model and loss
            else: 
                if hasattr(self.nn,'nn'): # check if neural network model has a nn attribute (single network)
                    gradient=tape.gradient(loss,self.nn.param) # compute the gradient using loss and neural network parameters
                else: # otherwise, assume neural network model does not have a nn attribute (actor-critic network)
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0]) # compute the actor gradient using loss and actor parameters
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1]) # compute the critic gradient using loss and critic parameters
            g_lock.release() # release the global lock for gradient computation
            if self.priority_flag==True and self.priority_p.value!=-1: # check if priority flag is True and priority index is not -1
                while True:
                    if p==self.priority_p.value: # check if current process index is equal to priority index
                        break # break the loop 
                    else:
                        continue # try again 
            lock[0].acquire() # acquire the lock for optimization operation
            if self.stop_func_(lock[0]): # check if stop flag is True
                return 0 # return 0 to indicate that optimization operation is stopped
            try:
                if hasattr(self.nn,'attenuate'): # check if neural network model has an attenuate attribute (custom attenuate function)
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # attenuate the gradient using neural network model, optimization counter, and process index
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p) # attenuate the actor gradient using neural network model, optimization counter, and process index
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p) # attenuate the critic gradient using neural network model, optimization counter, and process index
            except Exception as e:
                raise e # raise any exception
            try:
                try:
                    param=self.nn.opt(gradient) # update the neural network parameters using optimizer and gradient
                except Exception:
                    param=self.nn.opt(gradient,p) # update the neural network parameters using optimizer, gradient, and process index
            except Exception as e:
                raise e # raise any exception
            lock[0].release() # release the lock for optimization operation
        elif self.PO==3: # check if parallel optimization mode is 3
            if self.priority_flag==True and self.priority_p.value!=-1: # check if priority flag is True and priority index is not -1
                while True:
                    if p==self.priority_p.value: # check if current process index is equal to priority index
                        break # break the loop 
                    else:
                        continue # try again 
            if self.stop_func_(): # check if stop flag is True
                return 0 # return 0 to indicate that optimization operation is stopped
            if hasattr(self.nn,'gradient'): # check if neural network model has a gradient attribute (custom gradient function)
                gradient=self.nn.gradient(tape,loss) # compute the gradient using neural network model and loss
            else: 
                if hasattr(self.nn,'nn'): # check if neural network model has a nn attribute (single network)
                    gradient=tape.gradient(loss,self.nn.param) # compute the gradient using loss and neural network parameters
                else: # otherwise, assume neural network model does not have a nn attribute (actor-critic network)
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0]) # compute the actor gradient using loss and actor parameters
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1]) # compute the critic gradient using loss and critic parameters
            try:
                if hasattr(self.nn,'attenuate'): # check if neural network model has an attenuate attribute (custom attenuate function)
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # attenuate the gradient using neural network model, optimization counter, and process index
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p) # attenuate the actor gradient using neural network model, optimization counter, and process index
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p) # attenuate the critic gradient using neural network model, optimization counter, and process index
            except Exception as e:
                raise e # raise any exception
            try:
                try:
                    param=self.nn.opt(gradient) # update the neural network parameters using optimizer and gradient
                except Exception:
                    param=self.nn.opt(gradient,p) # update the neural network parameters using optimizer, gradient, and process index
            except Exception as e:
                raise e # raise any exception
        return loss,param # return the loss and the updated parameters
    
    
    def update_nn_param(self,param=None):
        if param==None: # check if param is None
            parameter_flat=nest.flatten(self.nn.param) # flatten the neural network parameters into a list
            parameter7_flat=nest.flatten(self.param[7]) # flatten the dictionary parameters into a list
        else: # otherwise, assume param is not None
            parameter_flat=nest.flatten(self.nn.param) # flatten the neural network parameters into a list
            parameter7_flat=nest.flatten(param) # flatten the param into a list
        for i in range(len(parameter_flat)): # loop over each element in the lists
            if param==None: # check if param is None
                state_ops.assign(parameter_flat[i],parameter7_flat[i]) # assign the value of dictionary parameter to neural network parameter
            else: # otherwise, assume param is not None
                state_ops.assign(parameter_flat[i],parameter7_flat[i]) # assign the value of param to neural network parameter
        self.nn.param=nest.pack_sequence_as(self.nn.param,parameter_flat) # pack the list back into the original structure of neural network parameters
        self.param[7]=nest.pack_sequence_as(self.param[7],parameter7_flat) # pack the list back into the original structure of dictionary parameters
        return
    
    
    def _train(self,p,j,batches,length,lock,g_lock):
        if j==batches-1: # check if this is the last batch of data
            index1=batches*self.batch # get the start index of the batch from the pool 
            index2=self.batch-(length-batches*self.batch) # get the end index of the batch from the pool 
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0) # concatenate the state data from two parts of the pool 
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0) # concatenate the action data from two parts of the pool 
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0) # concatenate the next state data from two parts of the pool 
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0) # concatenate the reward data from two parts of the pool 
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0) # concatenate the done flag data from two parts of the pool 
            if self.PO==2: # check if parallel optimization mode is 2
                if type(g_lock)!=list: # check if g_lock is not a list
                    pass # do nothing
                elif len(g_lock)==self.process: # check if g_lock has the same length as the number of processes
                    ln=p # assign the current process index to ln
                    g_lock=g_lock[ln] # assign the corresponding lock to g_lock
                else: # otherwise, assume g_lock has a different length from the number of processes
                    ln=int(np.random.choice(len(g_lock))) # randomly choose an index from g_lock
                    g_lock=g_lock[ln] # assign the corresponding lock to g_lock
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock) # execute the optimization operation using the batch data, process index, lock, and g_lock
            else: # otherwise, assume parallel optimization mode is not 2
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock) # execute the optimization operation using the batch data, process index, and lock
            self.param[7]=param # assign the updated parameters to the dictionary
            self.loss[p]+=loss # add the loss to the loss array of the current process
            if hasattr(self.nn,'bc'): # check if neural network model has a bc attribute (batch counter)
                bc=self.nn.bc[0] # get the batch counter from neural network model
                bc.assign_add(1) # increment the batch counter by 1
                self.nn.bc[0]=bc # assign the batch counter back to neural network model
        else: # otherwise, assume this is not the last batch of data
            index1=j*self.batch # get the start index of the batch from the pool 
            index2=(j+1)*self.batch # get the end index of the batch from the pool 
            state_batch=self.state_pool[p][index1:index2] # get the state data from the pool 
            action_batch=self.action_pool[p][index1:index2] # get the action data from the pool 
            next_state_batch=self.next_state_pool[p][index1:index2] # get the next state data from the pool 
            reward_batch=self.reward_pool[p][index1:index2] # get the reward data from the pool 
            done_batch=self.done_pool[p][index1:index2] # get the done flag data from the pool 
            if self.PO==2: # check if parallel optimization mode is 2
                if type(g_lock)!=list: # check if g_lock is not a list
                    pass # do nothing
                elif len(g_lock)==self.process: # check if g_lock has the same length as the number of processes
                    ln=p # assign the current process index to ln
                    g_lock=g_lock[ln] # assign the corresponding lock to g_lock
                else: # otherwise, assume g_lock has a different length from the number of processes
                    ln=int(np.random.choice(len(g_lock))) # randomly choose an index from g_lock
                    g_lock=g_lock[ln] # assign the corresponding lock to g_lock
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock) # execute the optimization operation using the batch data, process index, lock, and g_lock
            else: # otherwise, assume parallel optimization mode is not 2
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock) # execute the optimization operation using the batch data, process index, and lock
            self.param[7]=param # assign the updated parameters to the dictionary
            self.loss[p]+=loss # add the loss to the loss array of the current process
            if hasattr(self.nn,'bc'): # check if neural network model has a bc attribute (batch counter)
                bc=self.nn.bc[0] # get the batch counter from neural network model
                bc.assign_add(1) # increment the batch counter by 1
                self.nn.bc[0]=bc # assign the batch counter back to neural network model
        return
    
    
    def train_(self,p,lock,g_lock):
        if len(self.done_pool[p])<self.batch: # check if there are not enough data in the pool for training 
            return # return without training 
        else: # otherwise, assume there are enough data in the pool for training 
            self.loss[p]=0 # initialize the loss for current process as zero 
            length=len(self.done_pool[p]) # get the length of data in the pool 
            batches=int((length-length%self.batch)/self.batch) # compute how many batches are needed for training 
            if length%self.batch!=0: # check if there is a remainder of data in the pool 
                batches+=1 # increment the batches by 1
            for j in range(batches): # loop over each batch
                if self.priority_flag==True: # check if priority flag is True
                    self.priority_p.value=np.argmax(self.opt_counter) # assign the priority index as the process index with the highest optimization counter
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt: # check if maximum number of optimization steps per process per episode is set and reached
                        self.priority_p.value=int(self.priority_p.value) # convert the priority index to an integer
                    elif self.max_opt==None: # check if maximum number of optimization steps per process per episode is not set
                        self.priority_p.value=int(self.priority_p.value) # convert the priority index to an integer
                    else: # otherwise, assume maximum number of optimization steps per process per episode is set but not reached
                        self.priority_p.value=-1 # assign the priority index as -1 to indicate no priority
                if self.priority_flag==True: # check if priority flag is True
                    self.opt_counter[p]=0 # reset the optimization counter for current process as zero
                if hasattr(self.nn,'attenuate'): # check if neural network model has an attenuate attribute (custom attenuate function)
                    opt_counter=self.nn.opt_counter[0] # get the optimization counter from neural network model
                    opt_counter.scatter_update(tf.IndexedSlices(0,p)) # reset the optimization counter for current process as zero using scatter update operation
                    self.nn.opt_counter[0]=opt_counter # assign the optimization counter back to neural network model
                self._train(p,j,batches,length,lock,g_lock) # execute the training operation using the batch data, process index, lock, and g_lock
                if self.priority_flag==True: # check if priority flag is True
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i') # get the optimization counter array from shared memory
                    opt_counter+=1 # increment the optimization counter array by 1
                if hasattr(self.nn,'attenuate'): # check if neural network model has an attenuate attribute (custom attenuate function)
                    opt_counter=self.nn.opt_counter[0] # get the optimization counter from neural network model
                    opt_counter.assign(opt_counter+1) # increment the optimization counter by 1 using assign operation
                    self.nn.opt_counter[0]=opt_counter # assign the optimization counter back to neural network model
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[1].acquire() # acquire the lock for saving and updating operation 
            if self.update_step!=None: # check if update frequency for target network is set 
                if self.sc[p]%self.update_step==0: # check if step counter for current process reaches the update frequency
                    self.nn.update_param() # update the target network parameters using neural network model
            else: # otherwise, assume update frequency for target network is not set 
                self.nn.update_param() # update the target network parameters using neural network model
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[1].release() # release the lock for saving and updating operation 
            self.loss[p]=self.loss[p]/batches # compute the average loss for current process by dividing by batches
        self.sc[p]+=1 # increment the step counter for current process by 1
        if hasattr(self.nn,'ec'): # check if neural network model has an ec attribute (episode counter)
            ec=self.nn.ec[0] # get the episode counter from neural network model
            ec.assign_add(1) # increment the episode counter by 1 using assign operation
            self.nn.ec[0]=ec # assign the episode counter back to neural network model
        return
    
    
    def train(self,p,episode_count,lock,pool_lock,g_lock=None):
        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
            lock[1].acquire() # acquire the lock for initialization operation
        elif self.PO==3: # check if parallel optimization mode is 3
            lock[1].acquire() # acquire the lock for initialization operation
        self.state_pool[p]=None # initialize the state pool for current process as None
        self.action_pool[p]=None # initialize the action pool for current process as None
        self.next_state_pool[p]=None # initialize the next state pool for current process as None
        self.reward_pool[p]=None # initialize the reward pool for current process as None
        self.done_pool[p]=None # initialize the done pool for current process as None
        self.running_flag.append(1) # append a running flag of 1 to indicate that current process is running 
        self.process_counter.value+=1 # increment the process counter by 1
        self.finish_list.append(None) # append a finish flag of None to indicate that current process is not finished 
        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
            lock[1].release() # release the lock for initialization operation 
        elif self.PO==3: # check if parallel optimization mode is 3
            lock[1].release() # release the lock for initialization operation 
        try:
            epsilon=self.epsilon[p] # get the exploration rate for current process 
        except Exception:
            epsilon=None # set the exploration rate as None if exception occurs
        for k in range(episode_count): # loop over each episode 
            s=self.nn.env(p=p,initial=True) # reset the environment and get the initial state 
            if type(self.nn.param[0])!=list:
                s=np.array(s,self.nn.param[0].dtype.name) # convert the state to an array with the same data type as neural network parameters 
            else:
                s=np.array(s,self.nn.param[0][0].dtype.name) # convert the state to an array with the same data type as neural network parameters 
            if self.episode_step==None: # check if maximum number of steps per episode is not set 
                while True: # loop until episode ends 
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock) # execute an action and get the next state, reward, done flag, and index 
                    self.reward[p]+=r # add the reward to the reward array of current process 
                    s=next_s # update the state as next state 
                    if type(self.done_pool[p])==np.ndarray: # check if done pool is an array (enough data for training)
                        self.train_(p,lock,g_lock) # execute the training operation using current process index, lock, and g_lock 
                    if done: # check if episode ends 
                        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                            lock[1].acquire() # acquire the lock for episode operation 
                        self.total_episode.value+=1 # increment the total episode counter by 1
                        self.loss_list.append(self.loss[p]) # append the loss of current process to the loss list 
                        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                            lock[1].release() # release the lock for episode operation 
                        break # break the loop 
            else: # otherwise, assume maximum number of steps per episode is set 
                for l in range(self.episode_step): # loop over each step 
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock) # execute an action and get the next state, reward, done flag, and index 
                    self.reward[p]+=r # add the reward to the reward array of current process 
                    s=next_s # update the state as next state 
                    if type(self.done_pool[p])==np.ndarray: # check if done pool is an array (enough data for training)
                        self.train_(p,lock,g_lock) # execute the training operation using current process index, lock, and g_lock 
                    if done: # check if episode ends 
                        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                            lock[1].acquire() # acquire the lock for episode operation 
                        self.total_episode.value+=1 # increment the total episode counter by 1
                        self.loss_list.append(self.loss[p]) # append the loss of current process to the loss list 
                        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                            lock[1].release() # release the lock for episode operation 
                        break # break the loop 
                    if l==self.episode_step-1: # check if this is the last step of episode 
                        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                            lock[1].acquire() # acquire the lock for episode operation 
                        self.total_episode.value+=1 # increment the total episode counter by 1
                        self.loss_list.append(self.loss[p]) # append the loss of current process to the loss list 
                        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                            lock[1].release() # release the lock for episode operation 
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[1].acquire() # acquire the lock for saving and updating operation 
            elif len(lock)==3: # check if there are three locks (for saving and updating operation)
                lock[2].acquire() # acquire the third lock for saving and updating operation 
            if self.update_step!=None: # check if update frequency for target network is set 
                if self.sc[p]%self.update_step==0: # check if step counter for current process reaches the update frequency
                    self.nn.update_param() # update the target network parameters using neural network model
            else: # otherwise, assume update frequency for target network is not set 
                self.nn.update_param() # update the target network parameters using neural network model
            if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
                lock[1].release() # release the lock for saving and updating operation 
            elif len(lock)==3: # check if there are three locks (for saving and updating operation)
                lock[2].release() # release the third lock for saving and updating operation 
        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
            lock[1].acquire() # acquire the lock for saving and updating operation 
        elif len(lock)==3: # check if there are three locks (for saving and updating operation)
            lock[2].acquire() # acquire the third lock for saving and updating operation 
        self.save_() # execute the save_ function to save
        self.reward_list.append(self.reward[p]) # append the reward of current process to the reward list
        self.reward[p]=0 # reset the reward of current process as zero
        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
            lock[1].release() # release the lock for saving and updating operation 
        elif len(lock)==3: # check if there are three locks (for saving and updating operation)
            lock[2].release() # release the third lock for saving and updating operation 
        self.running_flag[p+1]=0 # set the running flag of current process as zero to indicate that it is not running anymore
        if p not in self.finish_list: # check if current process is not finished yet
            self.finish_list[p]=p # assign the process index to the finish list to indicate that it is finished 
        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
            lock[1].acquire() # acquire the lock for episode operation 
        elif self.PO==3: # check if parallel optimization mode is 3
            lock[1].acquire() # acquire the lock for episode operation 
        self.process_counter.value-=1 # decrement the process counter by 1
        if self.PO==1 or self.PO==2: # check if parallel optimization mode is 1 or 2
            lock[1].release() # release the lock for episode operation 
        elif self.PO==3: # check if parallel optimization mode is 3
            lock[1].release() # release the lock for episode operation 
        del self.state_pool[p] # delete the state pool of current process 
        del self.action_pool[p] # delete the action pool of current process 
        del self.next_state_pool[p] # delete the next state pool of current process 
        del self.reward_pool[p] # delete the reward pool of current process 
        del self.done_pool[p] # delete the done pool of current process 
        return # return to indicate that end operation is done
    
    
    def train_online(self,p,lock=None,g_lock=None):
        if hasattr(self.nn,'counter'): # check if neural network model has a counter attribute (training counter)
            self.nn.counter.append(0) # append a zero to indicate that current process has not started training yet
        while True: # loop until training stops
            if hasattr(self.nn,'save'): # check if neural network model has a save attribute (custom save function)
                self.nn.save(self.save,p) # save the neural network parameters using neural network model and process index
            if hasattr(self.nn,'stop_flag'): # check if neural network model has a stop_flag attribute
                if self.nn.stop_flag==True: # check if stop flag is True
                    return # return to indicate that training stops
            if hasattr(self.nn,'stop_func'): # check if neural network model has a stop_func attribute (custom stop function)
                if self.nn.stop_func(p): # check if stop function returns True for current process index
                    return # return to indicate that training stops
            if hasattr(self.nn,'suspend_func'): # check if neural network model has a suspend_func attribute (custom suspend function)
                self.nn.suspend_func(p) # execute the suspend function for current process index
            try:
                data=self.nn.online(p) # get the data for online training using neural network model and process index
            except Exception as e:
                self.nn.exception_list[p]=e # store the exception to the exception list of neural network model
            if data=='stop': # check if data is 'stop'
                return # return to indicate that training stops
            elif data=='suspend': # check if data is 'suspend'
                self.nn.suspend_func(p) # execute the suspend function for current process index
            try:
                if self.PO==2: # check if parallel optimization mode is 2
                    if type(g_lock)!=list: # check if g_lock is not a list
                        pass # do nothing
                    elif len(g_lock)==self.process: # check if g_lock has the same length as the number of processes
                        ln=p # assign the current process index to ln
                        g_lock=g_lock[ln] # assign the corresponding lock to g_lock
                    else: # otherwise, assume g_lock has a different length from the number of processes
                        ln=int(np.random.choice(len(g_lock))) # randomly choose an index from g_lock
                        g_lock=g_lock[ln] # assign the corresponding lock to g_lock
                    loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock,g_lock) # execute the optimization operation using the data, process index, lock, and g_lock
                else: # otherwise, assume parallel optimization mode is not 2
                    loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock) # execute the optimization operation using the data, process index, and lock
                self.param[7]=param # assign the updated parameters to the dictionary
            except Exception as e:
                if self.PO==1: # check if parallel optimization mode is 1
                    if lock[0].acquire(False): # try to acquire the lock for optimization operation 
                        lock[0].release() # release the lock for optimization operation 
                elif self.PO==2: # check if parallel optimization mode is 2
                    if g_lock.acquire(False): # try to acquire the global lock for gradient computation 
                        g_lock.release() # release the global lock for gradient computation 
                    if lock[0].acquire(False): # try to acquire the lock for optimization operation 
                        lock[0].release() # release the lock for optimization operation 
                self.nn.exception_list[p]=e # store the exception to the exception list of neural network model
            loss=loss.numpy() # convert the loss to a numpy array 
            if len(self.nn.train_loss_list)==self.nn.max_length: # check if train loss list reaches the maximum length 
                del self.nn.train_loss_list[0] # delete the oldest train loss from the list 
            self.nn.train_loss_list.append(loss) # append the train loss to the list 
            try:
                if hasattr(self.nn,'counter'): # check if neural network model has a counter attribute (training counter)
                    count=self.nn.counter[p] # get the training counter for current process 
                    count+=1 # increment the training counter by 1
                    self.nn.counter[p]=count # assign the training counter back to neural network model 
            except IndexError:
                self.nn.counter.append(0) # append a zero to indicate that current process has not started training yet 
                count=self.nn.counter[p] # get the training counter for current process 
                count+=1 # increment the training counter by 1
                self.nn.counter[p]=count # assign the training counter back to neural network model 
        return
