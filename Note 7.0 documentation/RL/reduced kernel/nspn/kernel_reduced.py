from tensorflow import function
from tensorflow import data as tf_data
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time


class kernel:
    def __init__(self,nn=None,save_episode=False):
        self.nn=nn # the neural network model to be trained
        try:
            self.nn.km=1 # a flag to indicate that the model is using kernel method
        except Exception:
            pass
        self.platform=None # the platform to use, such as TensorFlow or PyTorch
        self.state_pool=None # the pool of states
        self.action_pool=None # the pool of actions
        self.next_state_pool=None # the pool of next states
        self.reward_pool=None # the pool of rewards
        self.done_pool=None # the pool of done flags
        self.episode_set=[] # the list of episodes
        self.epsilon=None # the epsilon value for epsilon-greedy policy
        self.episode_step=None # the maximum number of steps per episode
        self.pool_size=None # the maximum size of the pool
        self.batch=None # the batch size for training
        self.update_step=None # the frequency of updating the network parameters
        self.trial_count=None # the number of trials to calculate the average reward
        self.criterion=None # the criterion for stopping the training
        self.reward_list=[] # the list of rewards per episode
        self.max_episode_count=None # the maximum number of episodes to save
        self.save_episode=save_episode # a flag to indicate whether to save episodes or not
        self.loss=None # the loss value for training
        self.loss_list=[] # the list of losses per episode
        self.sc=0 # a counter for steps
        self.total_episode=0 # a counter for episodes
        self.time=0 # a timer for training time
        self.total_time=0
    
    
    def action_vec(self):
        if self.epsilon!=None:
            self.action_one=np.ones(self.action_count,dtype=np.int8) # a vector of ones for action probabilities
        return
    
    
    def init(self):
        try:
            self.nn.pr.TD=np.array(0) # initialize the TD error for prioritized replay buffer
        except Exception as e:
            try:
               if self.nn.pr!=None: 
                   raise e
            except Exception:
                pass
        self.episode_set=[]
        self.state_pool=None 
        self.action_pool=None 
        self.next_state_pool=None 
        self.reward_pool=None 
        self.done_pool=None 
        self.reward_list=[]
        self.loss=0 
        self.loss_list=[]
        self.sc=0 
        self.total_episode=0 
        self.time=0 
        self.total_time=0 
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
        if epsilon!=None: 
            self.epsilon=epsilon # set the epsilon value for epsilon-greedy policy
        if episode_step!=None: 
            self.episode_step=episode_step # set the maximum number of steps per episode
        if pool_size!=None: 
            self.pool_size=pool_size # set the maximum size of the pool
        if batch!=None: 
            self.batch=batch # set the batch size for training
        if update_step!=None: 
            self.update_step=update_step # set the frequency of updating the network parameters
        if trial_count!=None: 
            self.trial_count=trial_count # set the number of trials to calculate the average reward
        if criterion!=None: 
            self.criterion=criterion # set the criterion for stopping the training
        self.action_vec() # create an action vector with the same length as the number of actions, and fill it with ones to indicate the initial probabilities of each action
        return
    
    
    def epsilon_greedy_policy(self,s):
        action_prob=self.action_one*self.epsilon/len(self.action_one) # initialize the action probabilities with epsilon
        try:
            best_a=np.argmax(self.nn.nn.fp(s)) # find the best action according to the network output
            action_prob[best_a]+=1-self.epsilon # increase the probability of the best action by 1-epsilon
        except Exception as e:
            try:
                if self.platform.DType!=None: # check if the platform is TensorFlow
                    raise e
            except Exception:
                best_a=self.nn.nn(s).argmax() # find the best action according to the network output
                action_prob[best_a.numpy()]+=1-self.epsilon # increase the probability of the best action by 1-epsilon
        return action_prob # return the action probabilities
    
    
    @function(jit_compile=True)
    def tf_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        with self.platform.GradientTape(persistent=True) as tape: # create a persistent gradient tape to record the gradients
            loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # calculate the loss function
        try:
            try:
                gradient=self.nn.gradient(tape,loss) # calculate the gradient using the custom gradient function
                try:
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param)) # apply the gradient to update the network parameters
                except Exception:
                    self.nn.opt(gradient) # apply the gradient to update the network parameters
            except Exception:
                try:
                    if self.nn.nn!=None: # check if the network is a single network or a pair of networks
                        gradient=tape.gradient(loss,self.nn.param) # calculate the gradient using the tape function
                        self.nn.opt.apply_gradients(zip(gradient,self.nn.param)) # apply the gradient to update the network parameters
                except Exception:
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0]) # calculate the gradient for the actor network
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1]) # calculate the gradient for the critic network
                    self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0])) # apply the gradient to update the actor network parameters
                    self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1])) # apply the gradient to update the critic network parameters
        except Exception as e:
            raise e
        return loss # return the loss value
    
    
    def pytorch_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # calculate the loss function
        self.nn.backward(loss) # calculate and accumulate the gradients using the custom backward function
        self.nn.opt() # apply the gradients to update the network parameters using the custom opt function
        return loss # return the loss value
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        try:
            if self.platform.DType!=None:  # check if the platform is TensorFlow
                loss=self.tf_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # use TensorFlow optimization method
        except Exception:
            loss=self.pytorch_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # use PyTorch optimization method
        return loss # return the loss value
    
    
    def pool(self,s,a,next_s,r,done):
        if type(self.state_pool)!=np.ndarray and self.state_pool==None: # check if the pool is empty or not
            self.state_pool=s # initialize the state pool with s
            if type(a)==int: # check if a is an integer or a vector
                if type(self.nn.param[0])!=list: 
                    a=np.array(a,self.nn.param[0].dtype.name) # convert a to an array with dtype matching nn.param[0]
                else:
                    a=np.array(a,self.nn.param[0][0].dtype.name) # convert a to an array with dtype matching nn.param[0][0]
                self.action_pool=np.expand_dims(a,axis=0) # initialize the action pool with a and add an extra dimension for batch size
            else:
                if type(self.nn.param[0])!=list: 
                    a=a.astype(self.nn.param[0].dtype.name) # convert a to an array with dtype matching nn.param[0]
                else:
                    a=a.astype(self.nn.param[0][0].dtype.name) # convert a to an array with dtype matching nn.param[0][0]
                self.action_poolself.action_pool=a # initialize the action pool with a
            self.next_state_pool=np.expand_dims(next_s,axis=0) # initialize the next state pool with next_s and add an extra dimension for batch size
            self.reward_pool=np.expand_dims(r,axis=0) # initialize the reward pool with r and add an extra dimension for batch size
            self.done_pool=np.expand_dims(done,axis=0) # initialize the done pool with done and add an extra dimension for batch size
        else:
            self.state_pool=np.concatenate((self.state_pool,s),0) # append s to the state pool along the first axis
            if type(a)==int: # check if a is an integer or a vector
                if type(self.nn.param[0])!=list: 
                    a=np.array(a,self.nn.param[0].dtype.name) # convert a to an array with dtype matching nn.param[0]
                else:
                    a=np.array(a,self.nn.param[0][0].dtype.name) # convert a to an array with dtype matching nn.param[0][0]
                self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0) # append a to the action pool along the first axis and add an extra dimension for batch size
            else:
                if type(self.nn.param[0])!=list: 
                    a=a.astype(self.nn.param[0].dtype.name) # convert a to an array with dtype matching nn.param[0]
                else:
                    a=a.astype(self.nn.param[0][0].dtype.name) # convert a to an array with dtype matching nn.param[0][0]
                self.action_pool=np.concatenate((self.action_pool,a),0) # append a to the action pool along the first axis
            self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(next_s,axis=0)),0) # append next_s to the next state pool along the first axis and add an extra dimension for batch size
            self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0) # append r to the reward pool along the first axis and add an extra dimension for batch size
            self.done_pool=np.concatenate((self.done_pool,np.expand_dims(done,axis=0)),0) # append done to the done pool along the first axis and add an extra dimension for batch size
        if len(self.state_pool)>self.pool_size: # check if the pool size exceeds the maximum size
            self.state_pool=self.state_pool[1:] # remove the oldest state from the state pool
            self.action_pool=self.action_pool[1:] # remove the oldest action from the action pool
            self.next_state_pool=self.next_state_pool[1:] # remove the oldest next state from the next state pool
            self.reward_pool=self.reward_pool[1:] # remove the oldest reward from the reward pool
            self.done_pool=self.done_pool[1:] # remove the oldest done flag from the done pool
        return
    
    
    def _train(self):
        if len(self.state_pool)<self.batch: # check if the pool size is smaller than the batch size
            return np.array(0.) # return zero as loss value
        else:
            loss=0 # initialize loss value as zero
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch) # calculate the number of batches to process
            if len(self.state_pool)%self.batch!=0: # check if there is any remainder after dividing by batch size
                batches+=1 # increase the number of batches by one
            try:
                for j in range(batches): # loop over each batch
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch) # get a batch of data using the custom data function
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # optimize the network parameters using the batch data and get the batch loss value
                    loss+=batch_loss # accumulate the loss value over batches
                    try:
                        try:
                            self.nn.bc.assign_add(1) # increase the batch counter by one using TensorFlow assign_add method
                        except Exception:
                            self.nn.bc+=1 # increase the batch counter by one using Python addition operator
                    except Exception:
                        pass
                if len(self.state_pool)%self.batch!=0: # check if there is any remainder after dividing by batch size
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch) # get a batch of data using the custom data function
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # optimize the network parameters using the batch data and get the batch loss value
                    loss+=batch_loss # accumulate the loss value over batches
                    try:
                        try:
                            self.nn.bc.assign_add(1) # increase the batch counter by one using TensorFlow assign_add method
                        except Exception:
                            self.nn.bc+=1 # increase the batch counter by one using Python addition operator
                    except Exception:
                        pass
            except Exception as e:
                try:
                    if self.nn.data_func!=None: # check if the custom data function is defined or not
                        raise e # raise the exception if it is defined
                except Exception:
                    try:
                        j=0 # initialize the batch index as zero
                        train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch) # create a TensorFlow dataset from the pool data and shuffle and batch it
                        for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds: # loop over each batch in the dataset
                            try:
                                if self.platform.DType!=None: # check if the platform is TensorFlow
                                    pass
                            except Exception:
                                state_batch=state_batch.numpy() # convert state_batch to numpy array
                                action_batch=action_batch.numpy() # convert action_batch to numpy array
                                next_state_batch=next_state_batch.numpy() # convert next_state_batch to numpy array
                                reward_batch=reward_batch.numpy() # convert reward_batch to numpy array
                                done_batch=done_batch.numpy() # convert done_batch to numpy array
                            batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # optimize the network parameters using the batch data and get the batch loss value
                            loss+=batch_loss # accumulate the loss value over batches
                            j+=1 # increase the batch index by one
                            try:
                                try:
                                    self.nn.bc.assign_add(1) # increase the batch counter by one using TensorFlow assign_add method
                                except Exception:
                                    self.nn.bc+=1 # increase the batch counter by one using Python addition operator
                            except Exception:
                                pass
                    except Exception as e:
                        raise e # raise the exception if any error occurs
            if self.update_step!=None: # check if the update frequency is defined or not
                if self.sc%self.update_step==0: # check if the step counter reaches the update frequency or not
                    self.nn.update_param() # update the network parameters using the custom update function
            else:
                self.nn.update_param() # update the network parameters using the custom update function
        try:
            if self.platform.DType!=None:  # check if the platform is TensorFlow
                loss=loss.numpy()/batches # convert loss to numpy array and divide it by batches to get the average loss value
        except Exception:
            loss=loss.detach().numpy()/batches # detach loss from computation graph and convert it to numpy array and divide it by batches to get the average loss value
        return loss # return the average loss value
    
    
    def train_(self):
        episode=[] # initialize an empty list for episode data
        self.reward=0 # initialize reward value as zero
        s=self.nn.env(initial=True) # get an initial state from the environment using the custom env function
        try:
            if self.platform.DType!=None:  # check if the platform is TensorFlow 
                if type(self.nn.param[0])!=list: 
                    s=np.array(s,self.nn.param[0].dtype.name) # convert s to an array with dtype matching nn.param[0]
                else:
                    s=np.array(s,self.nn.param[0][0].dtype.name) # convert s to an array with dtype matching nn.param[0][0]
        except Exception:
            pass 
        if self.episode_step==None:  # check if the maximum number of steps per episode is defined or not 
            while True:  # loop until done flag is True or an exception occurs 
                try:
                    try:
                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                            s=np.expand_dims(s,axis=0)  # adds=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                            if self.epsilon==None: # check if the epsilon value is defined or not
                                self.epsilon=self.nn.epsilon(self.sc) # calculate the epsilon value using the custom epsilon function
                            action_prob=self.epsilon_greedy_policy(s) # get the action probabilities using the epsilon-greedy policy function
                            a=np.random.choice(self.action_count,p=action_prob) # choose an action randomly according to the action probabilities
                    except Exception:
                        s=np.expand_dims(s,axis=0) # add an extra dimension for batch size to s
                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device) # convert s to a tensor with dtype matching platform.float and move it to the device
                        if self.epsilon==None: # check if the epsilon value is defined or not
                            self.epsilon=self.nn.epsilon(self.sc) # calculate the epsilon value using the custom epsilon function
                        action_prob=self.epsilon_greedy_policy(s) # get the action probabilities using the epsilon-greedy policy function
                        a=np.random.choice(self.action_count,p=action_prob) # choose an action randomly according to the action probabilities
                    next_s,r,done=self.nn.env(a) # get the next state, reward, and done flag from the environment using the custom env function and the chosen action
                    try:
                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                            if type(self.nn.param[0])!=list: 
                                next_s=np.array(next_s,self.nn.param[0].dtype.name) # convert next_s to an array with dtype matching nn.param[0]
                                r=np.array(r,self.nn.param[0].dtype.name) # convert r to an array with dtype matching nn.param[0]
                                done=np.array(done,self.nn.param[0].dtype.name) # convert done to an array with dtype matching nn.param[0]
                            else:
                                next_s=np.array(next_s,self.nn.param[0][0].dtype.name) # convert next_s to an array with dtype matching nn.param[0][0]
                                r=np.array(r,self.nn.param[0][0].dtype.name) # convert r to an array with dtype matching nn.param[0][0]
                                done=np.array(done,self.nn.param[0][0].dtype.name) # convert done to an array with dtype matching nn.param[0][0]
                    except Exception:
                        pass 
                except Exception as e:
                    try:
                       if self.nn.nn!=None:  # check if the network is a single network or a pair of networks 
                           raise e # raise the exception if it is a single network
                    except Exception:
                        try:
                            try:
                                if self.nn.action!=None:  # check if the custom action function is defined or not 
                                    try:
                                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                                            s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                            if self.epsilon==None:  # check if the epsilon value is defined or not 
                                                self.epsilon=self.nn.epsilon(self.sc)  # calculate the epsilon value using the custom epsilon function
                                            try:
                                                if self.nn.discriminator!=None:  # check if the custom discriminator function is defined or not 
                                                    a=self.nn.action(s)  # get the action using the custom action function
                                                    reward=self.nn.discriminator(s,a)  # get the reward using the custom discriminator function
                                                    s=np.squeeze(s)  # remove the extra dimension from s
                                            except Exception:
                                                a=self.nn.action(s).numpy()  # get the action using the custom action function and convert it to numpy array
                                    except Exception:
                                        s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)  # convert s to a tensor with dtype matching platform.float and move it to the device
                                        if self.epsilon==None:  # check if the epsilon value is defined or not 
                                            self.epsilon=self.nn.epsilon(self.sc)  # calculate the epsilon value using the custom epsilon function
                                        try:
                                            if self.nn.discriminator!=None:  # check if the custom discriminator function is defined or not 
                                                a=self.nn.action(s)  # get the action using the custom action function
                                                reward=self.nn.discriminator(s,a)  # get the reward using the custom discriminator function
                                                s=np.squeeze(s)  # remove the extra dimension from s
                                        except Exception:
                                            a=self.nn.action(s).detach().numpy()  # get the action using the custom action function and convert it to numpy array
                            except Exception:
                                try:
                                    if self.platform.DType!=None:  # check if the platform is TensorFlow 
                                        s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                        a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()  # get the action using the actor network and add some noise and convert it to numpy array
                                except Exception:
                                    s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)  # convert s to a tensor with dtype matching platform.float and move it to the device
                                    a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()  # get the action using the actor network and add some noise and convert it to numpy array
                        except Exception as e:
                            raise e # raise the exception if any error occurs
                    next_s,r,done=self.nn.env(a) # get the next state, reward, and done flag from the environment using the custom env function and the chosen action
                    try:
                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                            if type(self.nn.param[0])!=list: 
                                next_s=np.array(next_s,self.nn.param[0].dtype.name) # convert next_s to an array with dtype matching nn.param[0]
                                r=np.array(r,self.nn.param[0].dtype.name) # convert r to an array with dtype matching nn.param[0]
                                done=np.array(done,self.nn.param[0].dtype.name) # convert done to an array with dtype matching nn.param[0]
                            else:
                                next_s=np.array(next_s,self.nn.param[0][0].dtype.name) # convert next_s to an array with dtype matching nn.param[0][0]
                                r=np.array(r,self.nn.param[0][0].dtype.name) # convert r to an array with dtype matching nn.param[0][0]
                                done=np.array(done,self.nn.param[0][0].dtype.name) # convert done to an array with dtype matching nn.param[0][0]
                    except Exception:
                        pass 
                try:
                    self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,reward,done]) # add the data to the pool using the custom pool function
                except Exception as e:
                    try:
                        if self.nn.pool!=None: # check if the custom pool function is defined or not
                            raise e # raise the exception if it is defined
                    except Exception:
                        self.pool(s,a,next_s,r,done) # add the data to the pool using the default pool function
                try:
                    self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD) # append the initial TD error value to the TD error list for prioritized replay buffer
                    if len(self.state_pool)>self.pool_size: # check if the pool size exceeds the maximum size
                        TD=np.array(0) # create a zero TD error value
                        self.nn.pr.TD=np.append(TD,self.nn.pr.TD[2:]) # remove the oldest TD error value from the TD error list and append the zero TD error value
                except Exception as e:
                    try:
                        if self.nn.pr!=None: # check if the prioritized replay buffer is defined or not
                            raise e # raise the exception if it is defined
                    except Exception:
                        pass 
                self.reward=r+self.reward # accumulate the reward value over steps
                loss=self._train() # train the network using the pool data and get the loss value
                self.sc+=1 # increase the step counter by one
                if done: # check if done flag is True or not
                    if self.save_episode==True: # check if save episode flag is True or not
                        episode=[s,a,next_s,r] # create a list of data for this step
                    self.reward_list.append(self.reward) # append the reward value to the reward list
                    return loss,episode,done # return the loss value, episode data, and done flag
                elif self.save_episode==True: # check if save episode flag is True or not
                    episode=[s,a,next_s,r] # create a list of data for this step
                s=next_s # update s as next_s for next step 
        else: 
            for _ in range(self.episode_step): # loop over the maximum number of steps per episode
                try:
                    try:
                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                            s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                            if self.epsilon==None: # check if the epsilon value is defined or not
                                self.epsilon=self.nn.epsilon(self.sc) # calculate the epsilon value using the custom epsilon function
                            action_prob=self.epsilon_greedy_policy(s) # get the action probabilities using the epsilon-greedy policy function
                            a=np.random.choice(self.action_count,p=action_prob) # choose an action randomly according to the action probabilities
                    except Exception:
                        s=np.expand_dims(s,axis=0) # add an extra dimension for batch size to s
                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device) # convert s to a tensor with dtype matching platform.float and move it to the device
                        if self.epsilon==None: # check if the epsilon value is defined or not
                            self.epsilon=self.nn.epsilon(self.sc) # calculate the epsilon value using the custom epsilon function
                        action_prob=self.epsilon_greedy_policy(s) # get the action probabilities using the epsilon-greedy policy function
                        a=np.random.choice(self.action_count,p=action_prob) # choose an action randomly according to the action probabilities
                    next_s,r,done=self.nn.env(a) # get the next state, reward, and done flag from the environment using the custom env function and the chosen action
                    try:
                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                            if type(self.nn.param[0])!=list: 
                                next_s=np.array(next_s,self.nn.param[0].dtype.name) # convert next_s to an array with dtype matching nn.param[0]
                                r=np.array(r,self.nn.param[0].dtype.name) # convert r to an array with dtype matching nn.param[0]
                                done=np.array(done,self.nn.param[0].dtype.name) # convert done to an array with dtype matching nn.param[0]
                            else:
                                next_s=np.array(next_s,self.nn.param[0][0].dtype.name) # convert next_s to an array with dtype matching nn.param[0][0]
                                r=np.array(r,self.nn.param[0][0].dtype.name) # convert r to an array with dtype matching nn.param[0][0]
                                done=np.array(done,self.nn.param[0][0].dtype.name) # convert done to an array with dtype matching nn.param[0][0]
                    except Exception:
                        pass 
                except Exception as e:
                    try:
                       if self.nn.nn!=None:  # check if the network is a single network or a pair of networks 
                           raise e # raise the exception if it is a single network
                    except Exception:
                        try:
                            try:
                                if self.nn.action!=None:  # check if the custom action function is defined or not 
                                    try:
                                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                                            s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                            if self.epsilon==None:  # check if the epsilon value is defined or not 
                                                self.epsilon=self.nn.epsilon(self.sc)  # calculate the epsilon value using the custom epsilon function
                                            try:
                                                if self.nn.discriminator!=None:  # check if the custom discriminator function is defined or not 
                                                    a=self.nn.action(s)  # get the action using the custom action function
                                                    reward=self.nn.discriminator(s,a)  # get the reward using the custom discriminator function
                                                    s=np.squeeze(s)  # remove the extra dimension from s
                                            except Exception:
                                                a=self.nn.action(s).numpy()  # get the action using the custom action function and convert it to numpy array
                                    except Exception:
                                        s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)  # convert s to a tensor with dtype matching platform.float and move it to the device
                                        if self.epsilon==None:  # check if the epsilon value is defined or not 
                                            self.epsilon=self.nn.epsilon(self.sc)  # calculate the epsilon value using the custom epsilon function
                                        try:
                                            if self.nn.discriminator!=None:  # check if the custom discriminator function is defined or not 
                                                a=self.nn.action(s)  # get the action using thecustom action function
                                                reward=self.nn.discriminator(s,a)  # get the reward using the custom discriminator function
                                                s=np.squeeze(s)  # remove the extra dimension from s
                                        except Exception:
                                            a=self.nn.action(s).detach().numpy()  # get the action using the custom action function and convert it to numpy array
                            except Exception:
                                try:
                                    if self.platform.DType!=None:  # check if the platform is TensorFlow 
                                        s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                        a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()  # get the action using the actor network and add some noise and convert it to numpy array
                                except Exception:
                                    s=np.expand_dims(s,axis=0)  # add an extra dimension for batch size to s
                                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)  # convert s to a tensor with dtype matching platform.float and move it to the device
                                    a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()  # get the action using the actor network and add some noise and convert it to numpy array
                        except Exception as e:
                            raise e # raise the exception if any error occurs
                    next_s,r,done=self.nn.env(a) # get the next state, reward, and done flag from the environment using the custom env function and the chosen action
                    try:
                        if self.platform.DType!=None:  # check if the platform is TensorFlow 
                            if type(self.nn.param[0])!=list: 
                                next_s=np.array(next_s,self.nn.param[0].dtype.name) # convert next_s to an array with dtype matching nn.param[0]
                                r=np.array(r,self.nn.param[0].dtype.name) # convert r to an array with dtype matching nn.param[0]
                                done=np.array(done,self.nn.param[0].dtype.name) # convert done to an array with dtype matching nn.param[0]
                            else:
                                next_s=np.array(next_s,self.nn.param[0][0].dtype.name) # convert next_s to an array with dtype matching nn.param[0][0]
                                r=np.array(r,self.nn.param[0][0].dtype.name) # convert r to an array with dtype matching nn.param[0][0]
                                done=np.array(done,self.nn.param[0][0].dtype.name) # convert done to an array with dtype matching nn.param[0][0]
                    except Exception:
                        pass 
                try:
                    self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,reward,done]) # add the data to the pool using the custom pool function
                except Exception as e:
                    try:
                        if self.nn.pool!=None: # check if the custom pool function is defined or not
                            raise e # raise the exception if it is defined
                    except Exception:
                        self.pool(s,a,next_s,r,done) # add the data to the pool using the default pool function
                try:
                    self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD) # append the initial TD error value to the TD error list for prioritized replay buffer
                    if len(self.state_pool)>self.pool_size: # check if the pool size exceeds the maximum size
                        TD=np.array(0) # create a zero TD error value
                        self.nn.pr.TD=np.append(TD,self.nn.pr.TD[2:]) # remove the oldest TD error value from the TD error list and append the zero TD error value
                except Exception as e:
                    try:
                        if self.nn.pr!=None: # check if the prioritized replay buffer is defined or not
                            raise e # raise the exception if it is defined
                    except Exception:
                        pass 
                self.reward=r+self.reward # accumulate the reward value over steps
                loss=self._train() # train the network using the pool data and get the loss value
                self.sc+=1 # increase the step counter by one
                if done: # check if done flag is True or not
                    if self.save_episode==True: # check if save episode flag is True or not
                        episode=[s,a,next_s,r] # create a list of data for this step
                    self.reward_list.append(self.reward) # append the reward value to the reward list
                    return loss,episode,done # return the loss value, episode data, and done flag
                elif self.save_episode==True: # check if save episode flag is True or not
                    episode=[s,a,next_s,r] # create a list of data for this step
                s=next_s # update s as next_s for next step 
        self.reward_list.append(self.reward) # append the reward value to the reward list
        return loss,episode,done # return the loss value, episode data, and done flag
    
    
    def train(self,episode_count,save=None,one=True,p=None,s=None):
        avg_reward=None # initialize the average reward value as None
        if p==None: # check if p is defined or not
            self.p=9 # set p as 9 by default
        else:
            self.p=p-1 # decrease p by one
        if s==None: # check if s is defined or not
            self.s=1 # set s as 1 by default
            self.file_list=None # set file_list as None by default
        else:
            self.s=s-1 # decrease s by one
            self.file_list=[] # initialize an empty list for file_list
        if episode_count!=None: # check if episode_count is defined or not
            for i in range(episode_count): # loop over each episode
                t1=time.time() # record the start time of the episode
                loss,episode,done=self.train_() # train the network for one episode and get the loss value, episode data, and done flag
                if self.trial_count!=None: # check if trial_count is defined or not
                    if len(self.reward_list)>=self.trial_count: # check if the reward list has enough values for trial_count
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:]) # calculate the average reward value using the last trial_count values in the reward list
                        if self.criterion!=None and avg_reward>=self.criterion: # check if criterion is defined or not and if the average reward value meets the criterion or not
                            t2=time.time() # record the end time of the episode
                            self.total_time+=(t2-t1) # accumulate the total time over episodes
                            self._time=self.total_time-int(self.total_time) # get the decimal part of the total time
                            if self._time<0.5: 
                                self.total_time=int(self.total_time) # round down the total time if the decimal part is less than 0.5
                            else:
                                self.total_time=int(self.total_time)+1 # round up the total time if the decimal part is greater than or equal to 0.5
                            print('episode:{0}'.format(self.total_episode)) # print the total number of episodes
                            print('last loss:{0:.6f}'.format(loss)) # print the last loss value
                            print('average reward:{0}'.format(avg_reward)) # print the average reward value
                            print()
                            print('time:{0}s'.format(self.total_time)) # print the total time in seconds
                            return # stop training and return 
                self.loss=loss # update loss as the last loss value 
                self.loss_list.append(loss) # append loss to the loss list 
                self.total_episode+=1 # increase the total episode counter by one 
                if episode_count%10!=0: 
                    p=episode_count-episode_count%self.p 
                    p=int(p/self.p) 
                    s=episode_count-episode_count%self.s 
                    s=int(s/self.s) 
                else: 
                    p=episode_count/(self.p+1) 
                    p=int(p) 
                    s=episode_count/(self.s+1) 
                    s=int(s) 
                if p==0: 
                    p=1 
                if s==0: 
                    s=1 
                if i%p==0:  # check if i is a multiple of p or not 
                    if len(self.state_pool)>=self.batch:  # check if the pool size is larger than or equal to the batch size or not 
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))  # print the episode number and loss value 
                    if avg_reward!=None:  # check if avg_reward is None or not 
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))  # print the episode number and average reward value 
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))  # print the episode number and reward value 
                    print()
                if save!=None and i%s==0:  # check if save is None or not and if i is a multiple of s orif save!=None and i%s==0:  # check if save is None or not and if i is a multiple of s or not
                    self.save(self.total_episode,one) # save the network parameters and training data using the custom save function and the total episode number and one flag
                if self.save_episode==True: # check if save episode flag is True or not
                    if done: # check if done flag is True or not
                        episode.append('done') # append 'done' to the episode data list
                    self.episode_set.append(episode) # append the episode data list to the episode set list
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count: # check if max episode count is defined or not and if the episode set size reaches the max episode count or not
                        self.save_episode=False # set the save episode flag as False to stop saving episodes
                try:
                    try:
                        self.nn.ec.assign_add(1) # increase the episode counter by one using TensorFlow assign_add method
                    except Exception:
                        self.nn.ec+=1 # increase the episode counter by one using Python addition operator
                except Exception:
                    pass 
                t2=time.time() # record the end time of the episode
                self.time+=(t2-t1) # accumulate the time over episodes
        else: 
            i=0 # initialize the episode index as zero
            while True:  # loop until an exception occurs or the criterion is met 
                t1=time.time() # record the start time of the episode
                loss,episode,done=self.train_() # train the network for one episode and get the loss value, episode data, and done flag
                if self.trial_count!=None: # check if trial_count is defined or not
                    if len(self.reward_list)==self.trial_count: # check if the reward list has enough values for trial_count
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:]) # calculate the average reward value using the last trial_count values in the reward list
                        if avg_reward>=self.criterion: # check if the average reward value meets the criterion or not
                            t2=time.time() # record the end time of the episode
                            self.total_time+=(t2-t1) # accumulate the total time over episodes
                            self._time=self.total_time-int(self.total_time) # get the decimal part of the total time
                            if self._time<0.5: 
                                self.total_time=int(self.total_time) # round down the total time if the decimal part is less than 0.5
                            else:
                                self.total_time=int(self.total_time)+1 # round up the total time if the decimal part is greater than or equal to 0.5
                            print('episode:{0}'.format(self.total_episode)) # print the total number of episodes
                            print('last loss:{0:.6f}'.format(loss)) # print the last loss value
                            print('average reward:{0}'.format(avg_reward)) # print the average reward value
                            print()
                            print('time:{0}s'.format(self.total_time)) # print the total time in seconds
                            return # stop training and return 
                self.loss=loss # update loss as the last loss value 
                self.loss_list.append(loss) # append loss to the loss list 
                i+=1 # increase the episode index by one 
                self.total_episode+=1  # increase the total episode counter by one 
                if episode_count%10!=0: 
                    p=episode_count-episode_count%self.p 
                    p=int(p/self.p) 
                    s=episode_count-episode_count%self.s 
                    s=int(s/self.s) 
                else: 
                    p=episode_count/(self.p+1) 
                    p=int(p) 
                    s=episode_count/(self.s+1) 
                    s=int(s) 
                if p==0: 
                    p=1 
                if s==0: 
                    s=1 
                if i%p==0:  # check if i is a multiple of p or not 
                    if len(self.state_pool)>=self.batch:  # check if the pool size is larger than or equal to the batch size or not 
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))  # print the episode number and loss value 
                    if avg_reward!=None:  # check if avg_reward is None or not 
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))  # print the episode number and average reward valueelse:
                    else:  
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))  # print the episode number and reward value 
                    print()
                if save!=None and i%s==0:  # check if save is None or not and if i is a multiple of s or not
                    self.save(self.total_episode,one) # save the network parameters and training data using the custom save function and the total episode number and one flag
                if self.save_episode==True: # check if save episode flag is True or not
                    if done: # check if done flag is True or not
                        episode.append('done') # append 'done' to the episode data list
                    self.episode_set.append(episode) # append the episode data list to the episode set list
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count: # check if max episode count is defined or not and if the episode set size reaches the max episode count or not
                        self.save_episode=False # set the save episode flag as False to stop saving episodes
                try:
                    try:
                        self.nn.ec.assign_add(1) # increase the episode counter by one using TensorFlow assign_add method
                    except Exception:
                        self.nn.ec+=1 # increase the episode counter by one using Python addition operator
                except Exception:
                    pass 
                t2=time.time() # record the end time of the episode
                self.time+=(t2-t1) # accumulate the time over episodes
        self._time=self.time-int(self.time) # get the decimal part of the time
        if self._time<0.5: 
            self.total_time=int(self.time) # round down the time if the decimal part is less than 0.5
        else:
            self.total_time=int(self.time)+1 # round up the time if the decimal part is greater than or equal to 0.5
        self.total_time+=self.time # add the time to the total time 
        print('last loss:{0:.6f}'.format(loss)) # print the last loss value
        print('last reward:{0}'.format(self.reward)) # print the last reward value
        print()
        print('time:{0}s'.format(self.time)) # print the time in seconds
        return # return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1) # create a figure object with number 1
        plt.plot(np.arange(self.total_episode),self.reward_list) # plot the reward list against the episode number
        plt.xlabel('episode') # set the x-axis label as 'episode'
        plt.ylabel('reward') # set the y-axis label as 'reward'
        print('reward:{0:.6f}'.format(self.reward_list[-1])) # print the last reward value
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1) # create a figure object with number 1
        plt.plot(np.arange(self.total_episode),self.loss_list) # plot the loss list against the episode number
        plt.title('train loss') # set the title of the figure as 'train loss'
        plt.xlabel('episode') # set the x-axis label as 'episode'
        plt.ylabel('loss') # set the y-axis label as 'loss'
        print('loss:{0:.6f}'.format(self.loss_list[-1])) # print the last loss value
        return