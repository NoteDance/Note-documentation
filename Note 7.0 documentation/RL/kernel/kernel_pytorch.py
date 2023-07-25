import torch
from multiprocessing import Value,Array
from Note.nn.process.assign_device_pytorch import assign_device
import numpy as np
import statistics


class kernel:
    def __init__(self,nn=None,process=None,device='GPU'):
        self.nn=nn # the neural network model for the agent
        if process!=None:
            self.reward=np.zeros(process,dtype=np.float32) # the reward array for each process
            self.sc=np.zeros(process,dtype=np.int32) # the step counter array for each process
        self.device=device # the device to run the model on, either GPU or CPU
        self.state_pool={} # the dictionary to store the state transitions for each process
        self.action_pool={} # the dictionary to store the actions for each process
        self.next_state_pool={} # the dictionary to store the next states for each process
        self.reward_pool={} # the dictionary to store the rewards for each process
        self.done_pool={} # the dictionary to store the done flags for each process
        self.epsilon=None # the epsilon value for the epsilon-greedy policy
        self.episode_step=None # the maximum number of steps per episode
        self.pool_size=None # the maximum size of the state pool for each process
        self.batch=None # the batch size for training
        self.episode_=0 # the episode counter
        self.update_step=None # the frequency of updating the target network parameters
        self.trial_count=None # the number of trials to calculate the average reward
        self.process=process # the number of processes to run in parallel
        self.process_counter=0 # the counter of running processes
        self.probability_list=[] # the list of probabilities to select a process
        self.running_flag_list=[] # the list of flags to indicate whether a process is running or not
        self.finish_list=[] # the list of indices of finished processes
        self.running_flag=[] # the list of flags to indicate whether a process is running or not (shared)
        self.priority_flag=False # the flag to indicate whether to use priority-based optimization or not
        self.priority_p=0 # the index of the priority process to optimize first
        self.max_opt=None # the maximum number of optimization steps for a priority process
        self.stop=False # the flag to indicate whether to stop training or not
        self.save_flag=False # the flag to indicate whether to save the model parameters or not
        self.stop_flag=False # the flag to indicate whether to stop training or not (shared)
        self.opt_counter=None # the array to store the optimization counter for each process
        self.s=None # a temporary variable to store a state tensor
        self.filename='save.dat' # the file name to save and load the model parameters and states
        self.reward_list=[] # the list of rewards for each episode (shared)
        self.loss_list=[] # the list of losses for each episode (shared)
        self.total_episode=0 # the total number of episodes (shared)
    
    
    def init(self,manager):
        """This method is used to initialize some shared variables using a manager object"""
        
        self.state_pool=manager.dict(self.state_pool) 
        self.action_pool=manager.dict(self.action_pool)
        self.next_state_pool=manager.dict(self.next_state_pool)
        self.reward_pool=manager.dict(self.reward_pool)
        self.done_pool=manager.dict(self.done_pool)
        self.reward=Array('f',self.reward) 
        self.loss=np.zeros(self.process) 
        self.loss=Array('f',self.loss) 
        self.sc=Array('i',self.sc) 
        self.process_counter=Value('i',self.process_counter) 
        self.probability_list=manager.list(self.probability_list) 
        self.running_flag_list=manager.list(self.running_flag_list) 
        self.finish_list=manager.list(self.finish_list) 
        self.running_flag=manager.list([0]) 
        self.reward_list=manager.list(self.reward_list) 
        self.loss_list=manager.list(self.loss_list) 
        self.total_episode=Value('i',self.total_episode) 
        self.priority_p=Value('i',self.priority_p) 
        if self.priority_flag==True:
            self.opt_counter=Array('i',np.zeros(self.process,dtype=np.int32)) 
        try:
            self.nn.opt_counter=manager.list([self.nn.opt_counter])  
        except Exception:
            self.opt_counter_=manager.list() 
        try:
            self.nn.ec=manager.list([self.nn.ec])  
        except Exception:
            self.ec_=manager.list() 
        try:
            self.nn.bc=manager.list([self.nn.bc])
        except Exception:
            self.bc_=manager.list() 
        self.episode_=Value('i',self.total_episode.value) 
        self.stop_flag=Value('b',self.stop_flag) 
        self.save_flag=Value('b',self.save_flag) 
        self.file_list=manager.list([]) 
        return
    
    
    def init_online(self,manager):
        """This method is used to initialize some shared variables for online learning using a manager object"""
        
        self.nn.train_loss_list=manager.list([]) 
        self.nn.counter=manager.list([]) 
        self.nn.exception_list=manager.list([]) 
        return
    
    
    def action_vec(self):
        """This method is used to create a vector of ones with the same length as the action space"""
        
        self.action_one=np.ones(self.action_count,dtype=np.int8) 
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
        """This method is used to set up some hyperparameters for the agent"""
        
        if epsilon!=None:
            self.epsilon=np.ones(self.process)*epsilon # set the epsilon value for each process
        if episode_step!=None:
            self.episode_step=episode_step # set the maximum number of steps per episode
        if pool_size!=None:
            self.pool_size=pool_size # set the maximum size of the state pool for each process
        if batch!=None:
            self.batch=batch # set the batch size for training
        if update_step!=None:
            self.update_step=update_step # set the frequency of updating the target network parameters
        if trial_count!=None:
            self.trial_count=trial_count # set the number of trials to calculate the average reward
        if criterion!=None:
            self.criterion=criterion # set the criterion to stop training when the average reward reaches it
        if epsilon!=None:
            self.action_vec() # create a vector of ones with the same length as the action space
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon,p):
        """This method is used to implement an epsilon-greedy policy for action selection"""
        
        action_prob=self.action_one*epsilon/len(self.action_one) # initialize a uniform distribution over actions
        s=torch.tensor(s,dtype=torch.float).to(assign_device(p,self.device)) # convert the state to a tensor and assign it to a device
        best_a=self.nn.nn(s).argmax() # get the best action according to the network output
        action_prob[best_a.numpy()]+=1-epsilon # increase the probability of the best action by 1-epsilon
        return action_prob # return the action probability vector
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        """This method is used to store the state transitions in the state pool for a given process index"""
        
        pool_lock[index].acquire() # acquire a lock for the process index
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None: # if the state pool is empty for this index
                self.state_pool[index]=s # store the state as an array
                if type(a)==int: # if the action is an integer
                    a=np.array(a) # convert it to an array
                    self.action_pool[index]=np.expand_dims(a,axis=0) # store the action as an array with an extra dimension
                else: # if the action is already an array
                    self.action_pool[index]=a # store the action as it is
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0) # store the next state as an array with an extra dimension
                self.reward_pool[index]=np.expand_dims(r,axis=0) # store the reward as an array with an extra dimension
                self.done_pool[index]=np.expand_dims(done,axis=0) # store the done flag as an array with an extra dimension
            else: # if the state pool is not empty for this index
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0) # append the state to the existing state pool
                    if type(a)==int: # if the action is an integer
                        a=np.array(a) # convert it to an array
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0) # append the action to the existing action pool with an extra dimension
                    else: # if the action is already an array
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0) # append the action to the existing action pool as it is
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0) # append the next state to the existing next state pool with an extra dimension
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0) # append the reward to the existing reward pool with an extra dimension
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0) # append the done flag to the existing done pool with an extra dimension
                except Exception:
                    pass
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size: # if the state pool exceeds the maximum size for this index
                self.state_pool[index]=self.state_pool[index][1:] # remove the oldest state from the state pool
                self.action_pool[index]=self.action_pool[index][1:] # remove the oldest action from the action pool
                self.next_state_pool[index]=self.next_state_pool[index][1:] # remove the oldest next state from the next state pool
                self.reward_pool[index]=self.reward_pool[index][1:] # remove the oldest reward from the reward pool
                self.done_pool[index]=self.done_pool[index][1:] # remove the oldest done flag from the done pool
        except Exception:
            pool_lock[index].release() # release the lock for the process index
            return
        pool_lock[index].release() # release the lock for the process index
        return
    
    
    def get_index(self,p,lock):
        """This method is used to get a random index of a running process according to their probabilities"""
        
        while len(self.running_flag_list)<p: # wait until there are enough running flags in the list
            pass
        if len(self.running_flag_list)==p: # if there are exactly p running flags in the list
            lock[0].acquire() # acquire a lock for the shared variables
            self.running_flag_list.append(self.running_flag[1:].copy()) # append a copy of the running flag list (without the first element) to the list
            lock[0].release() # release the lock for the shared variables
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value: # if there are not enough or too many running flags for this process index
            self.running_flag_list[p]=self.running_flag[1:].copy() # update the running flag list for this process index with a copy of the running flag list (without the first element)
        while len(self.probability_list)<p: # wait until there are enough probabilities in the list
            pass
        if len(self.probability_list)==p: # if there are exactly p probabilities in the list
            lock[0].acquire() # acquire a lock for the shared variables
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])) # append a normalized array of probabilities based on the running flag list for this process index to the list
            lock[0].release() # release the lock for the shared variables
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]) # update the probability array for this process index based on the running flag list
        while True:
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p]) # randomly choose an index of a running process according to their probabilities
            if index in self.finish_list: # if this index is already finished
                continue # try again
            else: # if this index is not finished yet
                break # break out of the loop
        return index # return the chosen index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):
        """This method is used to interact with the environment and store the state transitions in the state pool"""
        
        if hasattr(self.nn,'nn'): # if there is a network model for the agent
            s=np.expand_dims(s,axis=0) # add an extra dimension to the state array
            action_prob=self.epsilon_greedy_policy(s,epsilon,p) # get the action probability vector according to the epsilon-greedy policy
            a=np.random.choice(self.action_count,p=action_prob) # randomly choose an action according to the action probability vector
        else: # if there is no network model for the agent
            if hasattr(self.nn,'action'): # if there is an action method for the agent
                s=np.expand_dims(s,axis=0) # add an extra dimension to the state array
                a=self.nn.action(s,p).numpy() # get the action array according to the action method
            else: # if there is no action method for the agent
                s=np.expand_dims(s,axis=0) # add an extra dimension to the state array
                a=(self.nn.actor(s)+self.nn.noise()).numpy() # get the action array according to the actor network and some noise
        next_s,r,done=self.nn.env(a,p) # get the next state, reward, and done flag from the environment
        index=self.get_index(p,lock) # get a random index of a running process according to their probabilities
        next_s=np.array(next_s) # convert the next state to an array
        r=np.array(r) # convert the reward to an array
        done=np.array(done) # convert the done flag to an array
        self.pool(s,a,next_s,r,done,pool_lock,index) # store the state transitions in the state pool for the chosen index
        return next_s,r,done,index # return the next state, reward, done flag, and index
    
    
    def end(self):
        """This method is used to check whether the training should be stopped or not"""
        
        if self.trial_count!=None: # if there is a trial count for calculating the average reward
            if len(self.reward_list)>=self.trial_count: # if there are enough rewards in the list
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:]) # calculate the average reward of the last trial count episodes
                if self.criterion!=None and avg_reward>=self.criterion: # if there is a criterion for stopping and the average reward reaches it
                    return True # return True to indicate that training should be stopped
        return False # return False to indicate that training should not be stopped
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p):
        """This method is used to optimize the network parameters using a batch of data"""
        
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p) # calculate the loss according to the network model and the data batch
        if self.priority_flag==True and self.priority_p.value!=-1: # if priority-based optimization is enabled and there is a priority process index
            while True:
                if self.stop_flag.value==True: # if training should be stopped (shared)
                    return None # return None to indicate that optimization is aborted
                if p==self.priority_p.value: # if this process index is equal to the priority process index
                    break # break out of the loop
                else: # if this process index is not equal to the priority process index
                    continue # try again
        if self.stop_func_(): # if training should be stopped (local)
            return None # return None to indicate that optimization is aborted
        loss=loss.clone() # clone the loss tensor to avoid modifying it in place
        self.nn.backward(loss,p) # perform backpropagation on the loss tensor according to the network model and this process index
        self.nn.opt(p) # perform optimization on the network parameters according to this process index
        return loss # return the loss tensor
    
    
    def _train(self,p,j,batches,length):
        """This method is used to train on a batch of data from the state pool for a given process index"""
        
        if j==batches-1: # if this is the last batch of data
            index1=batches*self.batch # get the starting index of this batch from the end of the state pool 
            index2=self.batch-(length-batches*self.batch) # get the ending index of this batch from the beginning of the state pool 
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0) # concatenate two slices of states from both ends of the state pool as a batch 
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0) # concatenate two slices of actions from both ends of the action pool as a batch 
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0) # concatenate two slices of next states from both ends of the next state pool as a batch 
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0) # concatenate two slices of rewards from both ends of the reward pool as a batch 
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0) # concatenate two slices of done flags from both ends of the done pool as a batch 
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p) # optimize the network parameters using this batch of data
            self.loss[p]+=loss # accumulate the loss for this process index
            if hasattr(self.nn,'bc'): # if there is a batch counter for the network model
                bc=self.nn.bc[0] # get the current value of the batch counter
                bc.assign_add(1) # increment the batch counter by 1
                self.nn.bc[0]=bc # update the batch counter for the network model
        else: # if this is not the last batch of data
            index1=j*self.batch # get the starting index of this batch from the state pool 
            index2=(j+1)*self.batch # get the ending index of this batch from the state pool 
            state_batch=self.state_pool[p][index1:index2] # get a slice of states from the state pool as a batch 
            action_batch=self.action_pool[p][index1:index2] # get a slice of actions from the action pool as a batch 
            next_state_batch=self.next_state_pool[p][index1:index2] # get a slice of next states from the next state pool as a batch 
            reward_batch=self.reward_pool[p][index1:index2] # get a slice of rewards from the reward pool as a batch 
            done_batch=self.done_pool[p][index1:index2] # get a slice of done flags from the done pool as a batch 
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p) # optimize the network parameters using this batch of data
            self.loss[p]+=loss # accumulate the loss for this process index
            if hasattr(self.nn,'bc'): # if there is a batch counter for the network model
                bc=self.nn.bc[0] # get the current value of the batch counter
                bc.assign_add(1) # increment the batch counter by 1
                self.nn.bc[0]=bc # update the batch counter for the network model
        return
    
    
    def train_(self,p,lock):
        """This method is used to train on all batches of data from the state pool for a given process index"""
        
        if len(self.done_pool[p])<self.batch: # if there are not enough data in the state pool for this process index
            return # return without training
        else: # if there are enough data in the state pool for this process index
            self.loss[p]=0 # reset the loss for this process index
            length=len(self.done_pool[p]) # get the length of the state pool for this process index
            batches=int((length-length%self.batch)/self.batch) # calculate the number of batches to train on
            if length%self.batch!=0: # if there is some remainder after dividing by the batch size
                batches+=1 # increment the number of batches by 1
            for j in range(batches): # for each batch
                if self.priority_flag==True: # if priority-based optimization is enabled
                    self.priority_p.value=np.argmax(self.opt_counter) # get the index of the process with the highest optimization counter as the priority process index
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt: # if there is a maximum number of optimization steps for a priority process and it is reached by the priority process index
                        self.priority_p.value=int(self.priority_p.value) # convert the priority process index to an integer type
                    elif self.max_opt==None: # if there is no maximum number of optimization steps for a priority process
                        self.priority_p.value=int(self.priority_p.value) # convert the priority process index to an integer type
                    else: # if there is a maximum number of optimization steps for a priority process and it is not reached by any process index
                        self.priority_p.value=-1 # set the priority process index to -1 to indicate no priority
                if self.priority_flag==True: # if priority-based optimization is enabled
                    self.opt_counter[p]=0 # reset the optimization counter for this process index
                if hasattr(self.nn,'attenuate'): # if there is an attenuate method for the network model
                    self.nn.attenuate(p) # attenuate some parameters according to the network model and this process index
                self._train(p,j,batches,length) # train on a batch of data from the state pool for this process index
                if self.priority_flag==True: # if priority-based optimization is enabled
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i') # get the optimization counter array as a numpy array
                    opt_counter+=1 # increment the optimization counter array by 1
                if hasattr(self.nn,'attenuate'): # if there is an attenuate method for the network model
                    opt_counter=self.nn.opt_counter[0] # get the current value of the optimization counter for the network model
                    opt_counter+=1 # increment the optimization counter for the network model by 1
                    self.nn.opt_counter[0]=opt_counter # update the optimization counter for the network model
            if self.update_step!=None: # if there is a frequency of updating the target network parameters
                if self.sc[p]%self.update_step==0: # if this process index reaches the update frequency
                    self.nn.update_param() # update the target network parameters according to the network model
            else: # if there is no frequency of updating the target network parameters
                self.nn.update_param() # update the target network parameters according to the network model
            self.loss[p]=self.loss[p]/batches # calculate the average loss for this process index
        self.sc[p]+=1 # increment the step counter for this process index
        if hasattr(self.nn,'ec'): # if there is an episode counter for the network model
            ec=self.nn.ec[0] # get the current value of the episode counter
            ec.assign_add(1) # increment the episode counter by 1
            self.nn.ec[0]=ec # update the episode counter for the network model
        return
    
    
    def train(self,p,episode_count,lock,pool_lock):
        """This method is used to start a process to train on multiple episodes"""
        
        lock[1].acquire() # acquire a lock for the shared variables
        self.state_pool[p]=None # initialize the state pool for this process index as None
        self.action_pool[p]=None # initialize the action pool for this process index as None
        self.next_state_pool[p]=None # initialize the next state pool for this process index as None
        self.reward_pool[p]=None # initialize the reward pool for this process index as None
        self.done_pool[p]=None # initialize the done pool for this process index as None
        self.running_flag.append(1) # append a 1 to the running flag list to indicate that this process is running 
        self.process_counter.value+=1 # increment the counter of running processes by 1 
        self.finish_list.append(None) # append a None to the finish list to indicate that this process is not finished yet 
        try:
            epsilon=self.epsilon[p] # get the epsilon value for this process index 
        except Exception:
            epsilon=None # set the epsilon value to None if there is no epsilon value 
        lock[1].release() # release the lock for the shared variables 
        for k in range(episode_count): # for each episode 
            s=self.nn.env(p=p,initial=True) # get an initial state from the environment according to this process index 
            s=np.array(s) # convert the state to an array 
            if self.episode_step==None: # if there is no maximum number of steps per episode 
                while True: # loop until done 
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock) # interact with the environment and store the state transitions in the state pool 
                    self.reward[p]+=r # accumulate the reward for this process index 
                    s=next_s # update the state with next state 
                    if type(self.done_pool[p])==np.ndarray: # if there are enough data in the state pool for this process index 
                        self.train_(p,lock) # train on all batches of data from the state pool for this process index 
                    if done: # if done flag is True 
                        if len(lock)==4: # if there is a fourth lock for saving and loading 
                            lock[3].acquire() # acquire a lock for saving and loading 
                        self.total_episode.value+=1 # increment the total number of episodes by 1 (shared)
                        self.loss_list.append(self.loss[p]) # append the loss for this process index to the loss list (shared)
                        if len(lock)==4: # if there is a fourth lock for saving and loading 
                            lock[3].acquire() # acquire a lock for saving and loading 
                        self.total_episode.value+=1 # increment the total number of episodes by 1 (shared)
                        self.loss_list.append(self.loss[p]) # append the loss for this process index to the loss list (shared)
                        if len(lock)==4: # if there is a fourth lock for saving and loading 
                            lock[3].release() # release the lock for saving and loading 
                        break # break out of the loop
            else: # if there is a maximum number of steps per episode 
                for l in range(self.episode_step): # for each step 
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock) # interact with the environment and store the state transitions in the state pool 
                    self.reward[p]+=r # accumulate the reward for this process index 
                    s=next_s # update the state with next state 
                    if type(self.done_pool[p])==np.ndarray: # if there are enough data in the state pool for this process index 
                        self.train_(p,lock) # train on all batches of data from the state pool for this process index 
                    if done: # if done flag is True 
                        if len(lock)==4: # if there is a fourth lock for saving and loading 
                            lock[3].acquire() # acquire a lock for saving and loading 
                        self.total_episode.value+=1 # increment the total number of episodes by 1 (shared)
                        self.loss_list.append(self.loss[p]) # append the loss for this process index to the loss list (shared)
                        if len(lock)==4: # if there is a fourth lock for saving and loading 
                            lock[3].release() # release the lock for saving and loading 
                        break # break out of the loop
                    if l==self.episode_step-1: # if this is the last step of the episode 
                        if len(lock)==4: # if there is a fourth lock for saving and loading 
                            lock[3].acquire() # acquire a lock for saving and loading 
                        self.total_episode.value+=1 # increment the total number of episodes by 1 (shared)
                        self.loss_list.append(self.loss[p]) # append the loss for this process index to the loss list (shared)
                        if len(lock)==4: # if there is a fourth lock for saving and loading 
                            lock[3].release() # release the lock for saving and loading 
            if len(lock)==3 or len(lock)==4: # if there is a third or fourth lock for saving and loading 
                lock[2].acquire() # acquire a lock for saving and loading 
            self.save_() # save the model parameters and states according to the network model
            self.reward_list.append(self.reward[p]) # append the reward for this process index to the reward list (shared)
            self.reward[p]=0 # reset the reward for this process index
            if len(lock)==3 or len(lock)==4: # if there is a third or fourth lock for saving and loading 
                lock[2].release() # release the lock for saving and loading
        self.running_flag[p+1]=0 # set the running flag to 0 to indicate that this process is not running anymore
        if p not in self.finish_list: # if this process index is not in the finish list yet
            self.finish_list[p]=p # add this process index to the finish list
        lock[1].acquire() # acquire a lock for the shared variables
        self.process_counter.value-=1 # decrement the counter of running processes by 1
        lock[1].release() # release the lock for the shared variables
        del self.state_pool[p] # delete the state pool for this process index
        del self.action_pool[p] # delete the action pool for this process index
        del self.next_state_pool[p] # delete the next state pool for this process index
        del self.reward_pool[p] # delete the reward pool for this process index
        del self.done_pool[p] # delete the done pool for this process index
        return
    
    
    def train_online(self,p,lock=None,g_lock=None):
        """This method is used to implement online learning using a single process"""
        
        if hasattr(self.nn,'counter'): # if there is a counter variable for the network model
            self.nn.counter.append(0) # append a 0 to the counter list
        while True: # loop until stopped
            if hasattr(self.nn,'save'): # if there is a save method for the network model
                self.nn.save(self.save,p) # save the model parameters and states according to the network model and this process index
            if hasattr(self.nn,'stop_flag'): # if there is a stop flag for the network model
                if self.nn.stop_flag==True: # if the stop flag is True
                    return # return to indicate that online learning is stopped
            if hasattr(self.nn,'stop_func'): # if there is a stop function for the network model
                if self.nn.stop_func(p): # if the stop function returns True according to this process index
                    return # return to indicate that online learning is stopped
            if hasattr(self.nn,'suspend_func'): # if there is a suspend function for the network model
                self.nn.suspend_func(p) # suspend some operations according to the network model and this process index
            try:
                data=self.nn.online(p) # get some data from the online source according to the network model and this process index
            except Exception as e: # if there is an exception raised
                self.nn.exception_list[p]=e # append the exception to the exception list for this process index
            if data=='stop': # if the data indicates to stop online learning
                return # return to indicate that online learning is stopped
            elif data=='suspend': # if the data indicates to suspend some operations
                self.nn.suspend_func(p) # suspend some operations according to the network model and this process index
            try:
                loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock,g_lock) # optimize the network parameters using the data and some locks
            except Exception as e: # if there is an exception raised
                self.nn.exception_list[p]=e # append the exception to the exception list for this process index
            loss=loss.numpy() # convert the loss tensor to a numpy array
            if len(self.nn.train_loss_list)==self.nn.max_length: # if the train loss list reaches its maximum length 
                del self.nn.train_loss_list[0] # delete the oldest element from the train loss list 
            self.nn.train_loss_list.append(loss) # append the loss to the train loss list 
            try:
                if hasattr(self.nn,'counter'): # if there is a counter variable for the network model 
                    count=self.nn.counter[p] # get the current value of the counter for this process index 
                    count+=1 # increment the counter by 1 
                    self.nn.counter[p]=count # update the counter for this process index 
            except IndexError: # if there is no counter for this process index yet 
                self.nn.counter.append(0) # append a 0 to the counter list 
                count=self.nn.counter[p] # get the current value of the counter for this process index 
                count+=1 # increment the counter by 1 
                self.nn.counter[p]=count # update the counter for this process index 
        return