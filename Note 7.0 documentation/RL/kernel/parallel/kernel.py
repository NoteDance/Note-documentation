import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
from multiprocessing import Value,Array
import numpy as np
import statistics


class kernel:
    def __init__(self,nn=None,process=None):
        self.nn=nn # the neural network model to be trained
        if process!=None: # the number of processes to be used for parallel training
            self.reward=np.zeros(process,dtype=np.float32) # the reward array for each process
            self.sc=np.zeros(process,dtype=np.int32) # the step counter array for each process
        self.epsilon=None # the epsilon value for epsilon-greedy policy
        self.episode_step=None # the maximum number of steps for each episode
        self.pool_size=None # the size of the memory pool for storing experience data
        self.episode=None # the number of episodes for training
        self.batch=None # the size of the batch for training
        self.update_step=None # the frequency of updating the neural network parameters
        self.trial_count=None # the number of trials to calculate the average reward
        self.criterion=None # the criterion for stopping the training based on the average reward
        self.process=process # the number of processes to be used for parallel training
        self.PO=None # the parallel optimization mode (1, 2 or 3)
        self.priority_flag=False # a flag indicating whether to use priority optimization or not
        self.max_opt=None # the maximum number of optimization steps for priority optimization
        self.stop=False # a flag indicating whether to stop the training or not
        self.opt_counter=None # a counter array for tracking the optimization steps for each process
        self.s=None # a state variable for online training
        self.filename='save.dat' # a file name for saving the neural network parameters
    
    
    def init(self,manager):
        self.state_pool=manager.dict({}) # a shared dictionary for storing state data for each process
        self.action_pool=manager.dict({}) # a shared dictionary for storing action data for each process
        self.next_state_pool=manager.dict({}) # a shared dictionary for storing next state data for each process
        self.reward_pool=manager.dict({}) # a shared dictionary for storing reward data for each process
        self.done_pool=manager.dict({}) # a shared dictionary for storing done flag data for each process
        self.reward=Array('f',self.reward) # a shared array for storing reward data for each process
        if type(self.nn.param[0])!=list:
            self.loss=np.zeros(self.process,dtype=self.nn.param[0].dtype.name) # a loss array for each process
        else:
            self.loss=np.zeros(self.process,dtype=self.nn.param[0][0].dtype.name) # a loss array for each process
        self.loss=Array('f',self.loss) # a shared array for storing loss data for each process
        self.sc=Array('i',self.sc) # a shared array for storing step counter data for each process
        self.process_counter=Value('i',0) # a shared value for counting the number of running processes
        self.probability_list=manager.list([]) # a shared list for storing probability data for each process
        self.running_flag_list=manager.list([]) # a shared list for storing running flag data for each process
        self.finish_list=manager.list([]) # a shared list for storing the finished process indices
        self.running_flag=manager.list([0]) # a shared list for storing the running flag for each process
        self.reward_list=manager.list([]) # a shared list for storing the total reward for each episode
        self.loss_list=manager.list([]) # a shared list for storing the average loss for each episode
        self.episode_counter=Value('i',0) # a shared value for counting the number of completed episodes
        self.total_episode=Value('i',0) # a shared value for counting the total number of episodes
        self.priority_p=Value('i',0) # a shared value for storing the priority process index
        if self.priority_flag==True: # if priority optimization is enabled
            self.opt_counter=Array('i',np.zeros(self.process,dtype=np.int32)) # a shared array for storing the optimization counter for each process
        try:
            self.nn.opt_counter=manager.list([self.nn.opt_counter])  # a shared list for storing the optimization counter for the neural network model 
        except Exception:
            self.opt_counter_=manager.list() # a dummy list for handling exception
        try:
            self.nn.ec=manager.list([self.nn.ec])  # a shared list for storing the episode counter for the neural network model 
        except Exception:
            self.ec_=manager.list() # a dummy list for handling exception
        try:
            self.nn.bc=manager.list([self.nn.bc]) # a shared list for storing the batch counter for the neural network model 
        except Exception:
            self.bc_=manager.list() # a dummy list for handling exception
        self.episode_=Value('i',self.total_episode.value) # a copy of the total episode value
        self.stop_flag=Value('b',False) # a shared flag for indicating whether to stop the training or not
        self.save_flag=Value('b',False) # a shared flag for indicating whether to save the neural network parameters or not
        self.file_list=manager.list([]) # a shared list for storing the file names for saving the neural network parameters
        self.param=manager.dict() # a shared dictionary for storing the neural network parameters
        self.param[7]=self.nn.param # assign the neural network parameters to the dictionary with key 7
        return
    
    
    def init_online(self,manager):
        self.nn.train_loss_list=manager.list([]) # a shared list for storing the training loss data for online training
        self.nn.counter=manager.list([]) # a shared list for storing the counter data for online training
        self.nn.exception_list=manager.list([]) # a shared list for storing the exception data for online training
        self.param=manager.dict() # a shared dictionary for storing the neural network parameters
        self.param[7]=self.nn.param # assign the neural network parameters to the dictionary with key 7
        return
    
    
    def action_vec(self):
        self.action_one=np.ones(self.action_count,dtype=np.int8) # an array of ones with length equal to the action count
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
        if epsilon!=None: # if epsilon value is given
            self.epsilon=np.ones(self.process)*epsilon # assign epsilon value to an array with length equal to the process count
            self.action_vec() # call the action_vec method to create an action array
        if episode_step!=None: # if episode step value is given
            self.episode_step=episode_step # assign episode step value to an attribute
        if pool_size!=None: # if pool size value is given
            self.pool_size=pool_size # assign pool size value to an attribute
        if batch!=None: # if batch size value is given
            self.batch=batch # assign batch size value to an attribute
        if update_step!=None: # if update step value is given
            self.update_step=update_step # assign update step value to an attribute
        if trial_count!=None: # if trial count value is given
            self.trial_count=trial_count # assign trial count value to an attribute
        if criterion!=None: # if criterion value is given
            self.criterion=criterion # assign criterion value to an attribute
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon):
        action_prob=self.action_one*epsilon/len(self.action_one) # calculate the action probability based on epsilon and action array
        best_a=np.argmax(self.nn.nn.fp(s)) # find the best action index based on the neural network output
        action_prob[best_a]+=1-epsilon # increase the probability of the best action by 1-epsilon
        return action_prob # return the action probability array
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        pool_lock[index].acquire() # acquire the lock for the given index
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None: # if the state pool for the given index is empty
                self.state_pool[index]=s # assign the state data to the state pool
                if type(a)==int: # if the action data is an integer
                    a=np.array(a) # convert it to a numpy array
                    self.action_pool[index]=np.expand_dims(a,axis=0) # add a dimension and assign it to the action pool
                else: # if the action data is not an integer
                    self.action_pool[index]=a # assign it to the action pool directly
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0) # add a dimension and assign the next state data to the next state pool
                self.reward_pool[index]=np.expand_dims(r,axis=0) # add a dimension and assign the reward data to the reward pool
                self.done_pool[index]=np.expand_dims(done,axis=0) # add a dimension and assign the done flag data to the done pool
            else: # if the state pool for the given index is not empty
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0) # concatenate the state data to the state pool
                    if type(a)==int: # if the action data is an integer
                        a=np.array(a) # convert it to a numpy array
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0) # add a dimension and concatenate it to the action pool
                    else: # if the action data is not an integer
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0) # concatenate it to the action pool directly
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0) # add a dimension and concatenate the next state data to the next state pool
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0) # add a dimension and concatenate the reward data to the reward pool
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0) # add a dimension and concatenate the done flag data to the done pool
                except Exception:
                    pass # ignore any exception that may occur during concatenation
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size: # if the state pool for the given index exceeds the pool size limit
                self.state_pool[index]=self.state_pool[index][1:] # remove the oldest state data from the state pool
                self.action_pool[index]=self.action_pool[index][1:] # remove the oldest action data from the action pool
                self.next_state_pool[index]=self.next_state_pool[index][1:] # remove the oldest next state data from the next state pool
                self.reward_pool[index]=self.reward_pool[index][1:] # remove the oldest reward data from the reward pool
                self.done_pool[index]=self.done_pool[index][1:] # remove the oldest done flag data from the done pool
        except Exception:
            pool_lock[index].release() # release the lock for the given index in case of any exception
            return 
        pool_lock[index].release() # release the lock for the given index after storing all data in pools 
        return
    
    
    def get_index(self,p,lock):
        while len(self.running_flag_list)<p: # wait until the running flag list has enough elements
            pass
        if len(self.running_flag_list)==p: # if the running flag list has exactly p elements
            if self.PO==1 or self.PO==2: # if the parallel optimization mode is 1 or 2
                lock[2].acquire() # acquire the lock with index 2
            elif self.PO==3: # if the parallel optimization mode is 3
                lock[0].acquire() # acquire the lock with index 0
            self.running_flag_list.append(self.running_flag[1:].copy()) # append a copy of the running flag list without the first element to the running flag list
            if self.PO==1 or self.PO==2: # if the parallel optimization mode is 1 or 2
                lock[2].release() # release the lock with index 2
            elif self.PO==3: # if the parallel optimization mode is 3
                lock[0].release() # release the lock with index 0
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value: # if the running flag list for the given p is not consistent with the process counter value
            self.running_flag_list[p]=self.running_flag[1:].copy() # update the running flag list for the given p with a copy of the running flag list without the first element
        while len(self.probability_list)<p: # wait until the probability list has enough elements
            pass
        if len(self.probability_list)==p: # if the probability list has exactly p elements
            if self.PO==1 or self.PO==2: # if the parallel optimization mode is 1 or 2
                lock[2].acquire() # acquire the lock with index 2
            elif self.PO==3: # if the parallel optimization mode is 3
                lock[0].acquire() # acquire the lock with index 0
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])) # append a normalized array of the running flag list for the given p to the probability list
            if self.PO==1 or self.PO==2: # if the parallel optimization mode is 1 or 2
                lock[2].release() # release the lock with index 2
            elif self.PO==3: # if the parallel optimization mode is 3
                lock[0].release() # release the lock with index 0
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]) # update the probability list for the given p with a normalized array of the running flag list for the given p 
        while True:
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p]) # randomly choose an index based on the probability list for the given p 
            if index in self.finish_list: # if the chosen index is in the finish list, meaning that process has finished training 
                continue # try again
            else: # otherwise, break out of the loop 
                break 
        return index # return the chosen index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):
        if hasattr(self.nn,'nn'): # if the neural network model has an attribute called nn 
            s=np.expand_dims(s,axis=0) # add a dimension to the state data 
            action_prob=self.epsilon_greedy_policy(s,epsilon) # calculate the action probability based on epsilon-greedy policy 
            a=np.random.choice(self.action_count,p=action_prob) # randomly choose an action based on action probability 
        else: # otherwise, assume that the neural network model does not have an attribute called nn 
            if hasattr(self.nn,'action'): # if the neural network model has an attribute called action 
                s=np.expand_dims(s,axis=0) # add a dimension to the state data 
                a=self.nn.action(s).numpy() # get an action from calling the action attribute of the neural network model and convert it to numpy array 
            else: # otherwise, assume that neither nn nor action attributes exist in the neural network model 
                s=np.expand_dims(s,axis=0) # add a dimension to the state data 
                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy() # get an action from calling the actor and noise attributes of the neural network model and convert it to numpy array 
        next_s,r,done=self.nn.env(a,p) # get next state, reward and done flag from calling env attribute of neural network model with action and process index as arguments 
        index=self.get_index(p,lock) # get an index from calling get_index
        if type(self.nn.param[0])!=list: # if the neural network model parameter is not a list
            next_s=np.array(next_s,self.nn.param[0].dtype.name) # convert the next state data to the same data type as the neural network model parameter
            r=np.array(r,self.nn.param[0].dtype.name) # convert the reward data to the same data type as the neural network model parameter
            done=np.array(done,self.nn.param[0].dtype.name) # convert the done flag data to the same data type as the neural network model parameter
        else: # otherwise, assume that the neural network model parameter is a list
            next_s=np.array(next_s,self.nn.param[0][0].dtype.name) # convert the next state data to the same data type as the first element of the neural network model parameter list
            r=np.array(r,self.nn.param[0][0].dtype.name) # convert the reward data to the same data type as the first element of the neural network model parameter list
            done=np.array(done,self.nn.param[0][0].dtype.name) # convert the done flag data to the same data type as the first element of the neural network model parameter list
        self.pool(s,a,next_s,r,done,pool_lock,index) # call the pool method to store all data in pools with lock and index as arguments
        return next_s,r,done,index # return next state, reward, done flag and index
    
    
    def end(self):
        if self.trial_count!=None: # if trial count value is given
            if len(self.reward_list)>=self.trial_count: # if reward list has enough elements
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:]) # calculate the average reward of the last trial count elements
                if self.criterion!=None and avg_reward>=self.criterion: # if criterion value is given and average reward is greater than or equal to criterion value
                    return True # return True, indicating that training should end
        return False # otherwise, return False, indicating that training should continue
    
    
    @tf.function(jit_compile=True) # use tensorflow function decorator with jit_compile option to improve computation efficiency and automatic differentiation
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock=None):
        with tf.GradientTape(persistent=True) as tape: # use tensorflow gradient tape with persistent option to record operations for automatic differentiation
            try:
                try:
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # calculate loss by calling loss attribute of neural network model with batch data as arguments
                except Exception:
                    loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p) # if exception occurs, try again with process index as an additional argument
            except Exception as e:
                raise e # if exception still occurs, raise it 
        if self.PO==1: # if parallel optimization mode is 1
            if self.priority_flag==True and self.priority_p.value!=-1: # if priority optimization is enabled and priority process index is not -1
                while True:
                    if self.stop_flag.value==True: # if stop flag is True
                        return None,None # return None values for loss and parameter
                    if p==self.priority_p.value: # if process index is equal to priority process index 
                        break # break out of the loop 
                    else: # otherwise 
                        continue # continue looping 
            lock[0].acquire() # acquire lock with index 0 
            if self.stop_func_(lock[0]): # call stop_func_ method with lock as argument and check if it returns True 
                return None,None # return None values for loss and parameter 
            if hasattr(self.nn,'gradient'): # if neural network model has an attribute called gradient 
                gradient=self.nn.gradient(tape,loss) # calculate gradient by calling gradient attribute of neural network model with tape and loss as arguments 
            else: # otherwise, assume that neural network model does not have an attribute called gradient 
                if hasattr(self.nn,'nn'): # if neural network model has an attribute called nn 
                    gradient=tape.gradient(loss,self.nn.param) # calculate gradient by calling gradient method of tape with loss and neural network model parameter as arguments 
                else: # otherwise, assume that neither nn nor gradient attributes exist in neural network model 
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0]) # calculate actor gradient by calling gradient method of tape with actor loss and actor parameter as arguments 
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1]) # calculate critic gradient by calling gradient method of tape with critic loss and critic parameter as arguments 
            try:
                if hasattr(self.nn,'attenuate'): # if neural network model has an attribute called attenuate 
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # attenuate gradient by calling attenuate attribute of neural network model with gradient, optimization counter and process index as arguments 
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p) # if exception occurs, try to attenuate actor gradient by calling attenuate attribute of neural network model with actor gradient, optimization counter and process index as arguments 
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p) # and try to attenuate critic gradient by calling attenuate attribute of neural network model with critic gradient, optimization counter and process index as arguments 
            except Exception as e:
                raise e # if exception still occurs, raise it 
            try:
                try:
                    param=self.nn.opt(gradient,p) # optimize parameter by calling opt attribute of neural network model with gradient and process index as arguments 
                except Exception:
                    param=self.nn.opt(gradient) # if exception occurs, try again without process index as argument 
            except Exception as e:
                raise e # if exception still occurs, raise it 
            lock[0].release() # release lock with index 0 
        elif self.PO==2: # if parallel optimization mode is 2
            g_lock.acquire() # acquire global lock 
            if self.stop_func_(g_lock): # call stop_func_ method with global lock as argument and check if it returns True 
                return None,None # return None values for loss and parameter 
            if hasattr(self.nn,'gradient'): # if neural network model has an attribute called gradient 
                gradient=self.nn.gradient(tape,loss) # calculate gradient by calling gradient attribute of neural network model with tape and loss as arguments 
            else: # otherwise, assume that neural network model does not have an attribute called gradient 
                if hasattr(self.nn,'nn'): # if neural network model has an attribute called nn 
                    gradient=tape.gradient(loss,self.nn.param) # calculate gradient by calling gradient method of tape with loss and neural network model parameter as arguments 
                else: # otherwise, assume that neither nn nor gradient attributes exist in neural network model 
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0]) # calculate actor gradient by calling gradient method of tape with actor loss and actor parameter as arguments 
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1]) # calculate critic gradient by calling gradient method of tape with critic loss and critic parameter as arguments 
            g_lock.release() # release global lock
            if self.priority_flag==True and self.priority_p.value!=-1: # if priority optimization is enabled and priority process index is not -1
                while True:
                    if self.stop_flag.value==True: # if stop flag is True
                        return None,None # return None values for loss and parameter
                    if p==self.priority_p.value: # if process index is equal to priority process index 
                        break # break out of the loop 
                    else: # otherwise 
                        continue # continue looping 
            lock[0].acquire() # acquire lock with index 0 
            try:
                if hasattr(self.nn,'attenuate'): # if neural network model has an attribute called attenuate 
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # attenuate gradient by calling attenuate attribute of neural network model with gradient, optimization counter and process index as arguments 
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p) # if exception occurs, try to attenuate actor gradient by calling attenuate attribute of neural network model with actor gradient, optimization counter and process index as arguments 
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p) # and try to attenuate critic gradient by calling attenuate attribute of neural network model with critic gradient, optimization counter and process index as arguments 
            except Exception as e:
                raise e # if exception still occurs, raise it 
            try:
                try:
                    param=self.nn.opt(gradient,p) # optimize parameter by calling opt attribute of neural network model with gradient and process index as arguments 
                except Exception:
                    param=self.nn.opt(gradient) # if exception occurs, try again without process index as argument 
            except Exception as e:
                raise e # if exception still occurs, raise it 
            lock[0].release() # release lock with index 0 
        elif self.PO==3: # if parallel optimization mode is 3
            if self.priority_flag==True and self.priority_p.value!=-1: # if priority optimization is enabled and priority process index is not -1
                while True:
                    if self.stop_flag.value==True: # if stop flag is True
                        return None,None # return None values for loss and parameter
                    if p==self.priority_p.value: # if process index is equal to priority process index 
                        break # break out of the loop 
                    else: # otherwise 
                        continue # continue looping 
            if self.stop_func_(): # call stop_func_ method without lock as argument and check if it returns True 
                return None,None # return None values for loss and parameter 
            if hasattr(self.nn,'gradient'): # if neural network model has an attribute called gradient 
                gradient=self.nn.gradient(tape,loss) # calculate gradient by calling gradient attribute of neural network model with tape and loss as arguments 
            else: # otherwise, assume that neural network model does not have an attribute called gradient 
                if hasattr(self.nn,'nn'): # if neural network model has an attribute called nn 
                    gradient=tape.gradient(loss,self.nn.param) # calculate gradient by calling gradient method of tape with loss and neural network model parameter as arguments 
                else: # otherwise, assume that neither nn nor gradient attributes exist in neural network model 
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0]) # calculate actor gradient by calling gradient method of tape with actor loss and actor parameter as arguments 
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1]) # calculate critic gradient by calling gradient method of tape with critic loss and critic parameter as arguments 
            try:
                if hasattr(self.nn,'attenuate'): # if neural network model has an attribute called attenuate 
                    try:
                        gradient=self.nn.attenuate(gradient,self.nn.opt_counter,p) # attenuate gradient by calling attenuate attribute of neural network model with gradient, optimization counter and process index as arguments 
                    except Exception:
                        actor_gradient=self.nn.attenuate(actor_gradient,self.nn.opt_counter,p) # if exception occurs, try to attenuate actor gradient by calling attenuate attribute of neural network model with actor gradient, optimization counter and process index as arguments 
                        critic_gradient=self.nn.attenuate(critic_gradient,self.nn.opt_counter,p) # and try to attenuate critic gradient by calling attenuate attribute of neural network model with critic gradient, optimization counter and process index as arguments 
            except Exception as e:
                raise e # if exception still occurs, raise it 
            try:
                try:
                    param=self.nn.opt(gradient,p) # optimize parameter by calling opt attribute of neural network model with gradient and process index as arguments 
                except Exception:
                    param=self.nn.opt(gradient) # if exception occurs, try again without process index as argument 
            except Exception as e:
                raise e # if exception still occurs, raise it 
        return loss,param # return loss and parameter
    
    
    def update_nn_param(self,param=None):
        if param==None: # if parameter is not given
            parameter_flat=nest.flatten(self.nn.param) # flatten the neural network model parameter to a list
            parameter7_flat=nest.flatten(self.param[7]) # flatten the parameter with key 7 in the shared dictionary to a list
        else: # otherwise, assume that parameter is given
            parameter_flat=nest.flatten(self.nn.param) # flatten the neural network model parameter to a list
            parameter7_flat=nest.flatten(param) # flatten the given parameter to a list
        for i in range(len(parameter_flat)): # loop through the indices of the flattened lists
            if param==None: # if parameter is not given
                state_ops.assign(parameter_flat[i],parameter7_flat[i]) # assign the value of the parameter with key 7 in the shared dictionary to the neural network model parameter
            else: # otherwise, assume that parameter is given
                state_ops.assign(parameter_flat[i],parameter7_flat[i]) # assign the value of the given parameter to the neural network model parameter
        self.nn.param=nest.pack_sequence_as(self.nn.param,parameter_flat) # pack the flattened list back to the original structure of the neural network model parameter
        self.param[7]=nest.pack_sequence_as(self.param[7],parameter7_flat) # pack the flattened list back to the original structure of the parameter with key 7 in the shared dictionary
        return
    
    
    def _train(self,p,j,batches,length,lock,g_lock):
        if j==batches-1: # if it is the last batch
            index1=batches*self.batch # calculate the start index of the batch
            index2=self.batch-(length-batches*self.batch) # calculate the end index of the batch
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0) # concatenate state data from two parts of the state pool to form a complete batch
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0) # concatenate action data from two parts of the action pool to form a complete batch
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0) # concatenate next state data from two parts of the next state pool to form a complete batch
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0) # concatenate reward data from two parts of the reward pool to form a complete batch
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0) # concatenate done flag data from two parts of the done pool to form a complete batch
            if self.PO==2: # if parallel optimization mode is 2
                if type(g_lock)!=list: # if global lock is not a list
                    pass # do nothing 
                elif len(g_lock)==self.process: # if global lock is a list with length equal to process count 
                    ln=p # set local lock index to process index 
                    g_lock=g_lock[ln] # set global lock to local lock with index ln 
                else: # otherwise, assume that global lock is a list with length not equal to process count 
                    ln=int(np.random.choice(len(g_lock))) # randomly choose a local lock index from global lock list 
                    g_lock=g_lock[ln] # set global lock to local lock with index ln 
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock) # call opt method with batch data, process index, lock and global lock as arguments and get loss and parameter as outputs 
            else: # otherwise, assume that parallel optimization mode is not 2 
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock) # call opt method with batch data, process index and lock as arguments and get loss and parameter as outputs 
            if self.stop_flag.value==True: # if stop flag is True 
                return # return without doing anything 
            self.param[7]=param # assign parameter to parameter with key 7 in shared dictionary 
            self.loss[p]+=loss # add loss to loss array for process index p 
            if hasattr(self.nn,'bc'): # if neural network model has an attribute called bc (batch counter)
                bc=self.nn.bc[0] # get bc value from neural network model 
                bc.assign_add(1) # increment bc value by 1 
                self.nn.bc[0]=bc # assign bc value back to neural network
        else: # otherwise, assume that it is not the last batch
            index1=j*self.batch # calculate the start index of the batch
            index2=(j+1)*self.batch # calculate the end index of the batch
            state_batch=self.state_pool[p][index1:index2] # get state data from state pool with start and end indices
            action_batch=self.action_pool[p][index1:index2] # get action data from action pool with start and end indices
            next_state_batch=self.next_state_pool[p][index1:index2] # get next state data from next state pool with start and end indices
            reward_batch=self.reward_pool[p][index1:index2] # get reward data from reward pool with start and end indices
            done_batch=self.done_pool[p][index1:index2] # get done flag data from done pool with start and end indices
            if self.PO==2: # if parallel optimization mode is 2
                if type(g_lock)!=list: # if global lock is not a list
                    pass # do nothing 
                elif len(g_lock)==self.process: # if global lock is a list with length equal to process count 
                    ln=p # set local lock index to process index 
                    g_lock=g_lock[ln] # set global lock to local lock with index ln 
                else: # otherwise, assume that global lock is a list with length not equal to process count 
                    ln=int(np.random.choice(len(g_lock))) # randomly choose a local lock index from global lock list 
                    g_lock=g_lock[ln] # set global lock to local lock with index ln 
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock,g_lock) # call opt method with batch data, process index, lock and global lock as arguments and get loss and parameter as outputs 
            else: # otherwise, assume that parallel optimization mode is not 2 
                loss,param=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p,lock) # call opt method with batch data, process index and lock as arguments and get loss and parameter as outputs 
            if self.stop_flag.value==True: # if stop flag is True 
                return # return without doing anything 
            self.param[7]=param # assign parameter to parameter with key 7 in shared dictionary 
            self.loss[p]+=loss # add loss to loss array for process index p 
            if hasattr(self.nn,'bc'): # if neural network model has an attribute called bc (batch counter)
                bc=self.nn.bc[0] # get bc value from neural network model 
                bc.assign_add(1) # increment bc value by 1 
                self.nn.bc[0]=bc # assign bc value back to neural network model 
        return # return without any value
    
    
    def train_(self,p,lock,g_lock):
        if len(self.done_pool[p])<self.batch: # if done pool for process index p has less elements than batch size
            return # return without doing anything 
        else: # otherwise, assume that done pool for process index p has enough elements for training
            self.loss[p]=0 # reset loss value for process index p to zero
            length=len(self.done_pool[p]) # get the length of the done pool for process index p
            batches=int((length-length%self.batch)/self.batch) # calculate the number of batches based on length and batch size
            if length%self.batch!=0: # if there is a remainder after dividing length by batch size
                batches+=1 # increment batches by 1
            for j in range(batches): # loop through the batch indices
                if self.priority_flag==True: # if priority optimization is enabled
                    self.priority_p.value=np.argmax(self.opt_counter) # set priority process index to the index of the maximum value in optimization counter array
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt: # if maximum optimization step value is given and optimization counter value for priority process index is greater than or equal to it
                        self.priority_p.value=int(self.priority_p.value) # convert priority process index to integer type
                    elif self.max_opt==None: # otherwise, assume that maximum optimization step value is not given
                        self.priority_p.value=int(self.priority_p.value) # convert priority process index to integer type
                    else: # otherwise, assume that maximum optimization step value is given but optimization counter value for priority process index is less than it
                        self.priority_p.value=-1 # set priority process index to -1, meaning no priority process
                if self.priority_flag==True: # if priority optimization is enabled
                    self.opt_counter[p]=0 # reset optimization counter value for process index p to zero
                if hasattr(self.nn,'attenuate'): # if neural network model has an attribute called attenuate
                    opt_counter=self.nn.opt_counter[0] # get opt_counter value from neural network model
                    opt_counter.scatter_update(tf.IndexedSlices(0,p)) # update opt_counter value with 0 at process index p
                    self.nn.opt_counter[0]=opt_counter # assign opt_counter value back to neural network model
                self._train(p,j,batches,length,lock,g_lock) # call _train method with process index, batch index, batch number, length, lock and global lock as arguments
                if self.stop_flag.value==True: # if stop flag is True 
                    return # return without doing anything 
                if self.priority_flag==True: # if priority optimization is enabled
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i') # get optimization counter array from shared memory
                    opt_counter+=1 # increment optimization counter array by 1
                if hasattr(self.nn,'attenuate'): # if neural network model has an attribute called attenuate
                    opt_counter=self.nn.opt_counter[0] # get opt_counter value from neural network model
                    opt_counter.assign(opt_counter+1) # increment opt_counter value by 1
                    self.nn.opt_counter[0]=opt_counter # assign opt_counter value back to neural network model
            if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                lock[1].acquire() # acquire lock with index 1
            if self.update_step!=None: # if update step value is given
                if self.sc[p]%self.update_step==0: # if step counter value for process index p is divisible by update step value
                    self.nn.update_param() # call update_param method of neural network model to update its parameters
            else: # otherwise, assume that update step value is not given
                self.nn.update_param() # call update_param method of neural network model to update its parameters
            if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                lock[1].release() # release lock with index 1
            self.loss[p]=self.loss[p]/batches # calculate the average loss for process index p by dividing the total loss by the number of batches 
        self.sc[p]+=1 # increment the step counter value for process index p by 1 
        if hasattr(self.nn,'ec'): # if neural network model has an attribute called ec (episode counter)
            ec=self.nn.ec[0] # get ec value from neural network model 
            ec.assign_add(1) # increment ec value by 1 
            self.nn.ec[0]=ec # assign ec value back to neural network model 
        return # return without any value
    
    
    def train(self,p,lock,pool_lock,g_lock=None):
        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
            lock[1].acquire() # acquire lock with index 1 
        elif self.PO==3: # if parallel optimization mode is 3 
            lock[1].acquire() # acquire lock with index 1 
        self.state_pool[p]=None # set state pool for process index p to None 
        self.action_pool[p]=None # set action pool for process index p to None 
        self.next_state_pool[p]=None # set next state pool for process index p to None 
        self.reward_pool[p]=None # set reward pool for process index p to None 
        self.done_pool[p]=None # set done pool for process index p to None 
        self.running_flag.append(1) # append 1 to running flag list, indicating that process is running 
        self.process_counter.value+=1 # increment process counter value by 1, indicating that one more process is running 
        self.finish_list.append(None) # append None to finish list, indicating that process is not finished yet 
        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
            lock[1].release() # release lock with index 1 
        elif self.PO==3: # if parallel optimization mode is 3 
            lock[1].release() # release lock with index 1 
        try:
            epsilon=self.epsilon[p] # get epsilon value for process index p from epsilon array 
        except Exception:
            epsilon=None # if exception occurs, set epsilon value to None 
        while True: # loop indefinitely until break condition is met 
            if self.stop_flag.value==True: # if stop flag is True 
                break # break out of the loop 
            if self.episode_counter.value>=self.episode: # if episode counter value is greater than or equal to episode
                break # break out of the loop 
            s=self.nn.env(p=p,initial=True) # get initial state from calling env attribute of neural network model with process index and initial flag as arguments 
            if type(self.nn.param[0])!=list: # if neural network model parameter is not a list
                s=np.array(s,self.nn.param[0].dtype.name) # convert state data to the same data type as the neural network model parameter
            else: # otherwise, assume that neural network model parameter is a list
                s=np.array(s,self.nn.param[0][0].dtype.name) # convert state data to the same data type as the first element of the neural network model parameter list
            if self.episode_step==None: # if episode step value is not given
                while True: # loop indefinitely until break condition is met 
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock) # get next state, reward, done flag and index from calling env method with state, epsilon, process index, lock and pool lock as arguments 
                    self.reward[p]+=r # add reward to reward array for process index p 
                    s=next_s # set state to next state 
                    if type(self.done_pool[p])==np.ndarray: # if done pool for process index p is a numpy array
                        self.train_(p,lock,g_lock) # call train_ method with process index, lock and global lock as arguments 
                        if self.stop_flag.value==True: # if stop flag is True 
                            break # break out of the loop 
                    if done: # if done flag is True 
                        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                            lock[1].acquire() # acquire lock with index 1 
                        elif len(lock)==4: # otherwise, if lock has length 4 
                            lock[3].acquire() # acquire lock with index 3 
                        self.episode_counter.value+=1 # increment episode counter value by 1, indicating that one more episode is completed 
                        self.total_episode.value+=1 # increment total episode value by 1, indicating that one more episode is completed 
                        self.loss_list.append(self.loss[p]) # append loss value for process index p to loss list 
                        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                            lock[1].release() # release lock with index 1 
                        elif len(lock)==4: # otherwise, if lock has length 4 
                            lock[3].release() # release lock with index 3 
                        break # break out of the loop 
            else: # otherwise, assume that episode step value is given
                for l in range(self.episode_step): # loop through the episode step indices
                    if self.episode_counter.value>=self.episode: # if episode counter value is greater than or equal to episode
                        break # break out of the loop 
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock) # get next state, reward, done flag and index from calling env method with state, epsilon, process index, lock and pool lock as arguments 
                    self.reward[p]+=r # add reward to reward array for process index p 
                    s=next_s # set state to next state 
                    if type(self.done_pool[p])==np.ndarray: # if done pool for process index p is a numpy array
                        self.train_(p,lock,g_lock) # call train_ method with process index, lock and global lock as arguments 
                        if self.stop_flag.value==True: # if stop flag is True 
                            break # break out of the loop 
                    if done: # if done flag is True 
                        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                            lock[1].acquire() # acquire lock with index 1 
                        elif len(lock)==4: # otherwise, if lock has length 4 
                            lock[3].acquire() # acquire lock with index 3 
                        self.episode_counter.value+=1 # increment episode counter value by 1, indicating that one more episode is completed 
                        self.total_episode.value+=1 # increment total episode value by 1, indicating that one more episode is completed 
                        self.loss_list.append(self.loss[p]) # append loss value for process index p to loss list 
                        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                            lock[1].release() # release lock with index 1 
                        elif len(lock)==4: # otherwise, if lock has length 4 
                            lock[3].release() # release lock with index 3 
                        break # break out of the loop 
                    if l==self.episode_step-1: # if it is the last episode step
                        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                            lock[1].acquire() # acquire lock with index 1 
                        elif len(lock)==4: # otherwise, if lock has length 4 
                            lock[3].acquire() # acquire lock with index 3 
                        self.episode_counter.value+=1 # increment episode counter value by 1, indicating that one more episode is completed 
                        self.total_episode.value+=1 # increment total episode value by 1, indicating that one more episode is completed 
                        self.loss_list.append(self.loss[p]) # append loss value for process index p to loss list 
                        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                            lock[1].release() # release lock with index 1 
                        elif len(lock)==4: # otherwise, if lock has length 4 
                            lock[3].release() # release lock with index 3 
            if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                lock[1].acquire() # acquire lock with index 1 
            elif len(lock)==3 or len(lock)==4: # otherwise, if lock has length 3 or 4
                lock[2].acquire() # acquire lock with index 2 
            self.save_() # call save_ method to save neural network parameters to file
            self.reward_list.append(self.reward[p]) # append reward value for process index p to reward list
            self.reward[p]=0 # reset reward value for process index p to zero
            if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
                lock[1].release() # release lock with index 1 
            elif len(lock)==3 or len(lock)==4: # otherwise, if lock has length 3 or 4
                lock[2].release() # release lock with index 2 
        self.running_flag[p+1]=0 # set running flag value for process index p plus one to zero, indicating that process is not running anymore
        if p not in self.finish_list: # if process index p is not in finish list
            self.finish_list[p]=p # assign process index p to finish list at process index p, indicating that process is finished
        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
            lock[1].acquire() # acquire lock with index 1 
        elif self.PO==3: # if parallel optimization mode is 3 
            lock[1].acquire() # acquire lock with index 1 
        self.process_counter.value-=1 # decrement process counter value by 1, indicating that one less process is running 
        if self.PO==1 or self.PO==2: # if parallel optimization mode is 1 or 2
            lock[1].release() # release lock with index 1 
        elif self.PO==3: # if parallel optimization mode is 3 
            lock[1].release() # release lock with index 1 
        del self.state_pool[p] # delete state pool for process index p
        del self.action_pool[p] # delete action pool for process index p
        del self.next_state_pool[p] # delete next state pool for process index p
        del self.reward_pool[p] # delete reward pool for process index p
        del self.done_pool[p] # delete done pool for process index p
        return # return without any value
    
    
    def train_online(self,p,lock=None,g_lock=None):
        if hasattr(self.nn,'counter'): # if neural network model has an attribute called counter
            self.nn.counter.append(0) # append zero to counter list of neural network model
        while True: # loop indefinitely until break condition is met 
            if hasattr(self.nn,'save'): # if neural network model has an attribute called save
                self.nn.save(self.save,p) # call save attribute of neural network model with save and process index as arguments
            if hasattr(self.nn,'stop_flag'): # if neural network model has an attribute called stop_flag
                if self.nn.stop_flag==True: # if stop_flag value of neural network model is True
                    return # return without doing anything
            if hasattr(self.nn,'stop_func'): # if neural network model has an attribute called stop_func
                if self.nn.stop_func(p): # call stop_func attribute of neural network model with process index as argument and check if it returns True
                    return # return without doing anything
            if hasattr(self.nn,'suspend_func'): # if neural network model has an attribute called suspend_func
                self.nn.suspend_func(p) # call suspend_func attribute of neural network model with process index as argument
            try:
                data=self.nn.online(p) # get data from calling online attribute of neural network model with process index as argument
            except Exception as e:
                self.nn.exception_list[p]=e # if exception occurs, assign it to exception list of neural network model at process index p
            if data=='stop': # if data is 'stop'
                return # return without doing anything
            elif data=='suspend': # if data is 'suspend'
                self.nn.suspend_func(p) # call suspend_func attribute of neural network model with process index as argument
            try:
                if self.PO==2: # if parallel optimization mode is 2
                    if type(g_lock)!=list: # if global lock is not a list
                        pass # do nothing 
                    elif len(g_lock)==self.process: # if global lock is a list with length equal to process count 
                        ln=p # set local lock index to process index 
                        g_lock=g_lock[ln] # set global lock to local lock with index ln 
                    else: # otherwise, assume that global lock is a list with length not equal to process count 
                        ln=int(np.random.choice(len(g_lock))) # randomly choose a local lock index from global lock list 
                        g_lock=g_lock[ln] # set global lock to local lock with index ln 
                    loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock,g_lock) # call opt method with data, process index, lock and global lock as arguments and get loss and parameter as outputs 
                else: # otherwise, assume that parallel optimization mode is not 2 
                    loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock) # call opt method with data, process index and lock as arguments and get loss and parameter as outputs 
            except Exception as e:
                if self.PO==1: # if parallel optimization mode is 1
                    if lock[0].acquire(False): # try to acquire lock with index 0 without blocking
                        lock[0].release() # release lock with index 0 
                elif self.PO==2: # if parallel optimization mode is 2
                    if g_lock.acquire(False): # try to acquire global lock without blocking
                        g_lock.release() # release global lock 
                    if lock[0].acquire(False): # try to acquire lock with index 0 without blocking
                        lock[0].release() # release lock with index 0 
                self.nn.exception_list[p]=e # assign exception to exception list of neural network model at process index p
            loss=loss.numpy() # convert loss to numpy array
            if len(self.nn.train_loss_list)==self.nn.max_length: # if train loss list of neural network model has reached the maximum length
                del self.nn.train_loss_list[0] # delete the first element of train loss list of neural network model
            self.nn.train_loss_list.append(loss) # append loss to train loss list of neural network model
            try:
                if hasattr(self.nn,'counter'): # if neural network model has an attribute called counter
                    count=self.nn.counter[p] # get count value from counter list of neural network model at process index p
                    count+=1 # increment count value by 1
                    self.nn.counter[p]=count # assign count value back to counter list of neural network model at process index p
            except IndexError:
                self.nn.counter.append(0) # if index error occurs, append zero to counter list of neural network model
                count=self.nn.counter[p] # get count value from counter list of neural network model at process index p
                count+=1 # increment count value by 1
                self.nn.counter[p]=count # assign count value back to counter list of neural network model at process index p
        return # return without any value
