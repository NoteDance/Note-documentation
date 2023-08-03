import torch
from multiprocessing import Value,Array
from Note.nn.parallel.assign_device_pytorch import assign_device
import numpy as np
import statistics


class kernel:
    def __init__(self,nn=None,process=None,device='GPU'):
        self.nn=nn # the neural network to be trained
        if process!=None:
            self.reward=np.zeros(process,dtype=np.float32) # the reward array for each process
            self.sc=np.zeros(process,dtype=np.int32) # the step counter array for each process
        self.device=device # the device to use, GPU or CPU
        self.epsilon=None # the epsilon value for epsilon-greedy policy
        self.episode_step=None # the maximum number of steps per episode
        self.pool_size=None # the size of the pool to store state, action, reward and done
        self.episode=None # the maximum number of episodes to run
        self.batch=None # the batch size for training
        self.update_step=None # the update step for target network
        self.trial_count=None # the number of trials to calculate average reward
        self.process=process # the number of processes to run in parallel
        self.priority_flag=False # the flag to indicate whether to use priority-based optimization
        self.max_opt=None # the maximum number of optimization steps per process
        self.stop=False # the flag to indicate whether to stop training
        self.s=None # the state variable for online training
        self.filename='save.dat' # the file name to save the model
    
    
    def init(self,manager):
        self.state_pool=manager.dict({}) # a dictionary to store state pool for each process
        self.action_pool=manager.dict({}) # a dictionary to store action pool for each process
        self.next_state_pool=manager.dict({}) # a dictionary to store next state pool for each process
        self.reward_pool=manager.dict({}) # a dictionary to store reward pool for each process
        self.done_pool=manager.dict({}) # a dictionary to store done pool for each process
        self.reward=Array('f',self.reward) # a shared array to store reward for each process
        self.loss=np.zeros(self.process) # an array to store loss for each process
        self.loss=Array('f',self.loss) # a shared array to store loss for each process
        self.sc=Array('i',self.sc) # a shared array to store step counter for each process
        self.process_counter=Value('i',0) # a shared value to count the number of running processes
        self.probability_list=manager.list([]) # a list to store probability distribution for selecting index from pool
        self.running_flag_list=manager.list([]) # a list to store running flag list for each process
        self.finish_list=manager.list([]) # a list to store finished processes' indices
        self.running_flag=manager.list([0]) # a list to store running flag for each process, initialized with 0
        self.reward_list=manager.list([]) # a list to store reward history 
        self.loss_list=manager.list([]) # a list to store loss history 
        self.episode_counter=Value('i',0) # a shared value to count the number of episodes completed by all processes 
        self.total_episode=Value('i',0) # a shared value to count the total number of episodes completed by all processes 
        self.priority_p=Value('i',0)  # a shared value to indicate which process has priority for optimization 
        if self.priority_flag==True: 
            self.opt_counter=Array('i',np.zeros(self.process,dtype=np.int32))  # an array to count optimization steps for each process 
        try:
            self.nn.opt_counter=manager.list([self.nn.opt_counter])  # a list to store optimization counter for each neural network 
        except Exception:
            self.opt_counter_=manager.list()  # an empty list if no optimization counter is available 
        try:
            self.nn.ec=manager.list([self.nn.ec])  # a list to store episode counter for each neural network 
        except Exception:
            self.ec_=manager.list()  # an empty list if no episode counter is available 
        try:
            self.nn.bc=manager.list([self.nn.bc])  # a list to store batch counter for each neural network 
        except Exception:
            self.bc_=manager.list()  # an empty list if no batch counter is available 
        self.episode_=Value('i',self.total_episode.value)  # a shared value to copy total episode value 
        self.stop_flag=Value('b',False)  # a shared value to indicate whether to stop training 
        self.save_flag=Value('b',False)  # a shared value to indicate whether to save the model 
        self.file_list=manager.list([])  # a list to store file names for saving the model 
        return
    
    
    def init_online(self,manager):
        self.nn.train_loss_list=manager.list([])  # a list to store training loss for online training 
        self.nn.counter=manager.list([])  # a list to store counter for online training 
        self.nn.exception_list=manager.list([])  # a list to store exceptions for online training 
        return
    
    
    def action_vec(self):
        self.action_one=np.ones(self.action_count,dtype=np.int8)  # an array of ones with the same size as action space 
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
        if epsilon!=None:
            self.epsilon=np.ones(self.process)*epsilon  # an array of epsilon values for each process 
            self.action_vec()  # create the action vector 
        if episode_step!=None:
            self.episode_step=episode_step  # set the maximum number of steps per episode 
        if pool_size!=None:
            self.pool_size=pool_size  # set the size of the pool 
        if batch!=None:
            self.batch=batch  # set the batch size for training 
        if update_step!=None:
            self.update_step=update_step  # set the update step for target network 
        if trial_count!=None:
            self.trial_count=trial_count  # set the number of trials to calculate average reward 
        if criterion!=None:
            self.criterion=criterion  # set the criterion for stopping training based on average reward 
        return
    
    
    def epsilon_greedy_policy(self,s,epsilon,p):
        action_prob=self.action_one*epsilon/len(self.action_one)  # create a uniform probability distribution for actions 
        s=torch.tensor(s,dtype=torch.float).to(assign_device(p,self.device))  # convert state to tensor and assign device 
        best_a=self.nn.nn(s).argmax()  # get the best action from neural network output 
        action_prob[best_a.numpy()]+=1-epsilon  # increase the probability of best action by (1-epsilon) 
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done,pool_lock,index):
        pool_lock[index].acquire()  # acquire lock for index
        try:
            if type(self.state_pool[index])!=np.ndarray and self.state_pool[index]==None:  # if state pool is empty
                self.state_pool[index]=s  # store state
                if type(a)==int:  
                    a=np.array(a)  
                    self.action_pool[index]=np.expand_dims(a,axis=0)  # store action with one dimension
                else:
                    self.action_pool[index]=a  # store action
                self.next_state_pool[index]=np.expand_dims(next_s,axis=0)  # store next state with one dimension
                self.reward_pool[index]=np.expand_dims(r,axis=0)  # store reward with one dimension
                self.done_pool[index]=np.expand_dims(done,axis=0)  # store done with one dimension
            else:  
                try:
                    self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)  # append state to state pool
                    if type(a)==int:
                        a=np.array(a)
                        self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)  # append action to action pool with one dimension
                    else:
                        self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)  # append action to action pool
                    self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)  # append next state to next state pool with one dimension
                    self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)  # append reward to reward pool with one dimension
                    self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)  # append done to done pool with one dimension
                except Exception:  
                    pass  
            if type(self.state_pool[index])==np.ndarray and len(self.state_pool[index])>self.pool_size:  # if state pool exceeds the size limit
                self.state_pool[index]=self.state_pool[index][1:]  # remove the oldest state from state pool
                self.action_pool[index]=self.action_pool[index][1:]  # remove the oldest action from action pool
                self.next_state_pool[index]=self.next_state_pool[index][1:]  # remove the oldest next state from next state pool
                self.reward_pool[index]=self.reward_pool[index][1:]  # remove the oldest reward from reward pool
                self.done_pool[index]=self.done_pool[index][1:]  # remove the oldest done from done pool
        except Exception:
            pool_lock[index].release()  # release lock for index if there is an exception
            return
        pool_lock[index].release()  # release lock for index after updating the pools
        return
    
    
    def get_index(self,p,lock):
        while len(self.running_flag_list)<p:  # wait until running flag list has enough elements
            pass
        if len(self.running_flag_list)==p:  # if running flag list has exactly p elements
            lock[0].acquire()  # acquire lock for running flag list
            self.running_flag_list.append(self.running_flag[1:].copy())  # copy the running flag list and append it to the list of lists
            lock[0].release()  # release lock for running flag list
        if len(self.running_flag_list[p])<self.process_counter.value or np.sum(self.running_flag_list[p])>self.process_counter.value:  # if running flag list has less elements than process counter or more elements than process counter
            self.running_flag_list[p]=self.running_flag[1:].copy()  # copy the running flag list and assign it to the p-th element of the list of lists
        while len(self.probability_list)<p:  # wait until probability list has enough elements
            pass
        if len(self.probability_list)==p:  # if probability list has exactly p elements
            lock[0].acquire()  # acquire lock for probability list
            self.probability_list.append(np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p]))  # calculate the probability distribution based on the running flag list and append it to the probability list
            lock[0].release()  # release lock for probability list
        self.probability_list[p]=np.array(self.running_flag_list[p],dtype=np.float16)/np.sum(self.running_flag_list[p])  # update the probability distribution based on the running flag list and assign it to the p-th element of the probability list
        while True:
            index=np.random.choice(len(self.probability_list[p]),p=self.probability_list[p])  # randomly choose an index from the probability distribution
            if index in self.finish_list:  # if index is in finish list, meaning that process has finished training
                continue  # try another index
            else:
                break  # break the loop and return the index
        return index
    
    
    def env(self,s,epsilon,p,lock,pool_lock):
        if hasattr(self.nn,'nn'):  # if neural network has a nn attribute, meaning that it is a Q-network or a policy network 
            s=np.expand_dims(s,axis=0)  # add one dimension to state 
            action_prob=self.epsilon_greedy_policy(s,epsilon,p)  # get the action probability from epsilon-greedy policy 
            a=np.random.choice(self.action_count,p=action_prob)  # randomly choose an action from action probability 
        else:  
            if hasattr(self.nn,'action'):  # if neural network has an action attribute, meaning that it is a deterministic policy network 
                s=np.expand_dims(s,axis=0)  # add one dimension to state 
                a=self.nn.action(s,p).numpy()  # get the action from neural network output 
            else:  
                s=np.expand_dims(s,axis=0)  # add one dimension to state 
                s=torch.tensor(s,dtype=torch.float).to(assign_device(p,self.device))  # convert state to tensor and assign device 
                a=(self.nn.actor(s)+self.nn.noise()).numpy()  # get the action from actor network output plus noise, meaning that it is a stochastic policy network 
        next_s,r,done=self.nn.env(a,p)  # get the next state, reward and done from environment with action and process index 
        index=self.get_index(p,lock)  # get an index from probability list with process index and lock 
        next_s=np.array(next_s)  
        r=np.array(r)
        done=np.array(done)
        self.pool(s,a,next_s,r,done,pool_lock,index)  # update the pools with state, action, next state, reward, done, pool lock and index 
        return next_s,r,done,index
    
    
    def end(self):
        if self.trial_count!=None:  # if trial count is not None
            if len(self.reward_list)>=self.trial_count:  # if reward list has enough elements
                avg_reward=statistics.mean(self.reward_list[-self.trial_count:])  # calculate the average reward of the last trial count elements
                if self.criterion!=None and avg_reward>=self.criterion:  # if criterion is not None and average reward is greater than or equal to criterion
                    return True  # return True, meaning that training should end
        return False  # otherwise, return False, meaning that training should continue
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch,p):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)  # calculate the loss from batch data and process index 
        if self.priority_flag==True and self.priority_p.value!=-1:  # if priority flag is True and priority process index is not -1
            while True:
                if self.stop_flag.value==True:  # if stop flag is True
                    return None  # return None, meaning that optimization should stop
                if p==self.priority_p.value:  # if process index is equal to priority process index
                    break  # break the loop and continue optimization
                else:
                    continue  # otherwise, continue the loop and wait for priority
        if self.stop_func_():  # if stop function returns True
            return None  # return None, meaning that optimization should stop
        loss=loss.clone()  # clone the loss tensor 
        self.nn.backward(loss,p)  # perform backward propagation with loss and process index 
        self.nn.opt(p)  # perform optimization with process index 
        return loss
    
    
    def _train(self,p,j,batches,length):
        if j==batches-1:  # if it is the last batch
            index1=batches*self.batch  # get the start index of the batch 
            index2=self.batch-(length-batches*self.batch)  # get the end index of the batch 
            state_batch=np.concatenate((self.state_pool[p][index1:length],self.state_pool[p][:index2]),0)  # get the state batch by concatenating the last part and the first part of state pool 
            action_batch=np.concatenate((self.action_pool[p][index1:length],self.action_pool[p][:index2]),0)  # get the action batch by concatenating the last part and the first part of action pool 
            next_state_batch=np.concatenate((self.next_state_pool[p][index1:length],self.next_state_pool[p][:index2]),0)  # get the next state batch by concatenating the last part and the first part of next state pool 
            reward_batch=np.concatenate((self.reward_pool[p][index1:length],self.reward_pool[p][:index2]),0)  # get the reward batch by concatenating the last part and the first part of reward pool 
            done_batch=np.concatenate((self.done_pool[p][index1:length],self.done_pool[p][:index2]),0)  # get the done batch by concatenating the last part and the first part of done pool 
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)  # get the loss from opt method with batch data and process index 
            self.loss[p]+=loss  # add loss to loss array for process index 
            if hasattr(self.nn,'bc'):  
                bc=self.nn.bc[0]  
                bc.assign_add(1)  
                self.nn.bc[0]=bc  
        else:  
            index1=j*self.batch  
            index2=(j+1)*self.batch  
            state_batch=self.state_pool[p][index1:index2]  
            action_batch=self.action_pool[p][index1:index2]  
            next_state_batch=self.next_state_pool[p][index1:index2]  
            reward_batch=self.reward_pool[p][index1:index2]  
            done_batch=self.done_pool[p][index1:index2]  
            loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch,p)  
            self.loss[p]+=loss  
            if hasattr(self.nn,'bc'):  
                bc=self.nn.bc[0]  
                bc.assign_add(1)  
                self.nn.bc[0]=bc  
        return
    
    
    def train_(self,p):
        if len(self.done_pool[p])<self.batch:  # if done pool has less elements than batch size
            return  # return without training
        else:
            self.loss[p]=0  # reset loss for process index to zero
            length=len(self.done_pool[p])  # get the length of done pool
            batches=int((length-length%self.batch)/self.batch)  # get the number of batches based on length and batch size
            if length%self.batch!=0:  # if length is not divisible by batch size
                batches+=1  # add one more batch
            for j in range(batches):  # loop over batches
                if self.priority_flag==True:  # if priority flag is True
                    self.priority_p.value=np.argmax(self.opt_counter)  # get the priority process index by finding the maximum of opt counter array
                    if self.max_opt!=None and self.opt_counter[self.priority_p.value]>=self.max_opt:  # if max opt is not None and opt counter for priority process index is greater than or equal to max opt
                        self.priority_p.value=int(self.priority_p.value)  # convert priority process index to integer
                    elif self.max_opt==None:  # if max opt is None
                        self.priority_p.value=int(self.priority_p.value)  # convert priority process index to integer
                    else:
                        self.priority_p.value=-1  # set priority process index to -1, meaning no priority
                if self.priority_flag==True:
                    self.opt_counter[p]=0  # reset opt counter for process index to zero
                if hasattr(self.nn,'attenuate'):  
                    opt_counter=self.nn.opt_counter[0]  
                    opt_counter[p]=0  
                    self.nn.opt_counter[0]=opt_counter  
                self._train(p,j,batches,length)  # call _train method with process index, batch index, number of batches and length of done pool 
                if self.priority_flag==True:
                    opt_counter=np.frombuffer(self.opt_counter.get_obj(),dtype='i')  # get the opt counter array from shared memory 
                    opt_counter+=1  # increment the opt counter array by one 
                if hasattr(self.nn,'attenuate'):  
                    opt_counter=self.nn.opt_counter[0]  
                    opt_counter+=1  
                    self.nn.opt_counter[0]=opt_counter  
            if self.update_step!=None:  # if update step is not None
                if self.sc[p]%self.update_step==0:  # if step counter for process index is divisible by update step
                    self.nn.update_param()  # update the target network parameters 
            else:
                self.nn.update_param()  # update the target network parameters 
            self.loss[p]=self.loss[p]/batches  # calculate the average loss for process index by dividing by number of batches 
        self.sc[p]+=1  # increment the step counter for process index by one 
        if hasattr(self.nn,'ec'):  
            ec=self.nn.ec[0]  
            ec.assign_add(1)  
            self.nn.ec[0]=ec  
        return
    
    
    def train(self,p,lock,pool_lock):
        lock[1].acquire()  # acquire lock for running flag list 
        self.state_pool[p]=None  # initialize state pool for process index to None 
        self.action_pool[p]=None  # initialize action pool for process index to None 
        self.next_state_pool[p]=None  # initialize next state pool for process index to None 
        self.reward_pool[p]=None  # initialize reward pool for process index to None 
        self.done_pool[p]=None  # initialize done pool for process index to None 
        self.running_flag.append(1)  # append one to running flag list, meaning that process is running 
        self.process_counter.value+=1  # increment the process counter by one 
        self.finish_list.append(None)  # append None to finish list, meaning that process is not finished 
        lock[1].release()  # release lock for running flag list 
        try:
            epsilon=self.epsilon[p]  # get the epsilon value for process index 
        except Exception:
            epsilon=None  # set epsilon value to None if there is an exception 
        while True:
            if self.stop_flag.value==True:  # if stop flag is True
                break  # break the loop and stop training
            if self.episode!=None and self.episode_counter.value>=self.episode:  # if episode limit is not None and episode counter is greater than or equal to episode limit
                break  # break the loop and stop training
            s=self.nn.env(p=p,initial=True)  # get the initial state from environment with process index 
            s=np.array(s)  
            if self.episode_step==None:  # if episode step limit is None
                while True:
                    if self.episode!=None and self.episode_counter.value>=self.episode:  # if episode limit is not None and episode counter is greater than or equal to episode limit
                        break  # break the loop and stop training
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)  # get the next state, reward, done and index from env method with state, epsilon, process index, lock and pool lock 
                    self.reward[p]+=r  # add reward to reward array for process index 
                    s=next_s  # assign next state to state 
                    if type(self.done_pool[p])==np.ndarray:  # if done pool is not empty
                        self.train_(p)  # call train_ method with process index 
                        if self.stop_flag.value==True:  # if stop flag is True
                            break  # break the loop and stop training
                    if done:  # if done is True, meaning that episode is finished
                        if len(lock)==4:  # if there are four locks
                            lock[3].acquire()  # acquire the fourth lock for episode counter 
                        self.episode_counter.value+=1  # increment the episode counter by one 
                        self.total_episode.value+=1  # increment the total episode by one 
                        self.loss_list.append(self.loss[p])  # append the loss for process index to loss list 
                        if len(lock)==4:  # if there are four locks
                            lock[3].release()  # release the fourth lock for episode counter 
                        break  # break the loop and start a new episode
            else:  # if episode step limit is not None
                for l in range(self.episode_step):  # loop over episode step limit
                    if self.episode!=None and self.episode_counter.value>=self.episode:  # if episode limit is not None and episode counter is greater than or equal to episode limit
                        break  # break the loop and stop training
                    next_s,r,done,index=self.env(s,epsilon,p,lock,pool_lock)  # get the next state, reward, done and index from env method with state, epsilon, process index, lock and pool lock 
                    self.reward[p]+=r  # add reward to reward array for process index 
                    s=next_s  # assign next state to state 
                    if type(self.done_pool[p])==np.ndarray:  # if done pool is not empty
                        self.train_(p)  # call train_ method with process index 
                        if self.stop_flag.value==True:  # if stop flag is True
                            break  # break the loop and stop training
                    if done:  # if done is True, meaning that episode is finished
                        if len(lock)==4:  # if there are four locks
                            lock[3].acquire()  # acquire the fourth lock for episode counter 
                        self.episode_counter.value+=1  # increment the episode counter by one 
                        self.total_episode.value+=1  # increment the total episode by one 
                        self.loss_list.append(self.loss[p])  # append the loss for process index to loss list 
                        if len(lock)==4:  # if there are four locks
                            lock[3].release()  # release the fourth lock for episode counter 
                        break  # break the loop and start a new episode
                    if l==self.episode_step-1:  # if it is the last step of episode
                        if len(lock)==4:  # if there are four locks
                            lock[3].acquire()  # acquire the fourth lock for episode counter 
                        self.episode_counter.value+=1
                        self.total_episode.value+=1  # increment the total episode by one 
                        self.loss_list.append(self.loss[p])  # append the loss for process index to loss list 
                        if len(lock)==4:  # if there are four locks
                            lock[3].release()  # release the fourth lock for episode counter 
            if len(lock)==3 or len(lock)==4:  # if there are three or four locks
                lock[2].acquire()  # acquire the third lock for saving model 
            self.save_()  # call save_ method to save the model 
            self.reward_list.append(self.reward[p])  # append the reward for process index to reward list 
            self.reward[p]=0  # reset the reward for process index to zero 
            if len(lock)==3 or len(lock)==4:  # if there are three or four locks
                lock[2].release()  # release the third lock for saving model 
        self.running_flag[p+1]=0  # set the running flag for process index to zero, meaning that process is not running 
        if p not in self.finish_list:  # if process index is not in finish list
            self.finish_list[p]=p  # add process index to finish list, meaning that process is finished 
        lock[1].acquire()  # acquire lock for running flag list 
        self.process_counter.value-=1  # decrement the process counter by one 
        lock[1].release()  # release lock for running flag list 
        del self.state_pool[p]  # delete state pool for process index 
        del self.action_pool[p]  # delete action pool for process index 
        del self.next_state_pool[p]  # delete next state pool for process index 
        del self.reward_pool[p]  # delete reward pool for process index 
        del self.done_pool[p]  # delete done pool for process index 
        return
    
    
    def train_online(self,p,lock=None,g_lock=None):
        if hasattr(self.nn,'counter'):  
            self.nn.counter.append(0)  
        while True:
            if hasattr(self.nn,'save'):  
                self.nn.save(self.save,p)  
            if hasattr(self.nn,'stop_flag'):  
                if self.nn.stop_flag==True:  
                    return  
            if hasattr(self.nn,'stop_func'):  
                if self.nn.stop_func(p):  
                    return  
            if hasattr(self.nn,'suspend_func'):  
                self.nn.suspend_func(p)  
            try:
                data=self.nn.online(p)  # get the online data from neural network with process index 
            except Exception as e:
                self.nn.exception_list[p]=e  # store the exception in exception list for process index 
            if data=='stop':  # if data is 'stop', meaning that online training should stop
                return  
            elif data=='suspend':  # if data is 'suspend', meaning that online training should suspend
                self.nn.suspend_func(p)  # call suspend function with process index 
            try:
                loss,param=self.opt(data[0],data[1],data[2],data[3],data[4],p,lock,g_lock)  # get the loss and parameter from opt method with data, process index, lock and global lock 
            except Exception as e:
                self.nn.exception_list[p]=e  # store the exception in exception list for process index 
            loss=loss.numpy()  
            if len(self.nn.train_loss_list)==self.nn.max_length:  # if train loss list has reached the maximum length
                del self.nn.train_loss_list[0]  # delete the oldest element from train loss list
            self.nn.train_loss_list.append(loss)  # append the loss to train loss list
            try:
                if hasattr(self.nn,'counter'):  
                    count=self.nn.counter[p]  
                    count+=1  
                    self.nn.counter[p]=count  
            except IndexError:
                self.nn.counter.append(0)  
                count=self.nn.counter[p]  
                count+=1  
                self.nn.counter[p]=count  
        return
