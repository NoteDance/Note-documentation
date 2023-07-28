import tensorflow as tf
import numpy as np
import statistics
import time


# This is a class for kernel-based reinforcement learning
class kernel:
    # This is the constructor method
    def __init__(self,nn=None,save_episode=False):
        self.nn=nn # This is the neural network model
        if hasattr(self.nn,'km'): # If the model has a kernel matrix attribute
            self.nn.km=1 # Set it to 1
        self.platform=None # This is the platform for the model, such as tensorflow or pytorch
        self.state_pool=None # This is the pool for storing states
        self.action_pool=None # This is the pool for storing actions
        self.next_state_pool=None # This is the pool for storing next states
        self.reward_pool=None # This is the pool for storing rewards
        self.done_pool=None # This is the pool for storing done flags
        self.episode_set=[] # This is the list for storing episodes
        self.epsilon=None # This is the epsilon value for epsilon-greedy policy
        self.episode_step=None # This is the maximum number of steps per episode
        self.pool_size=None # This is the maximum size of the pool
        self.batch=None # This is the batch size for training
        self.update_step=None # This is the frequency of updating the network parameters
        self.trial_count=None # This is the number of trials for calculating average reward
        self.criterion=None # This is the criterion for stopping training when average reward reaches it
        self.reward_list=[] # This is the list for storing rewards per episode
        self.suspend=False # This is a flag for suspending training
        self.save_epi=None # This is a flag for saving episodes
        self.max_episode_count=None # This is the maximum number of episodes to save
        self.save_episode=save_episode # This is a flag for saving episodes or not
        self.filename='save.dat' # This is the filename for saving data
        self.loss=None # This is the loss value for training
        self.loss_list=[] # This is the list for storing loss values per episode
        self.sc=0 # This is a counter for steps
        self.total_episode=0 # This is a counter for episodes
        self.time=0 # This is a timer for training time
        self.total_time=0 # This is a timer for total time
    
    
    def action_vec(self): # This is a method for creating action probability vector
        if self.epsilon!=None: # If epsilon value is not None
            self.action_one=np.ones(self.action_count,dtype=np.int8) # Create a vector of ones with length equal to action count
            return
    
    
    def init(self): # This is a method for initializing some attributes
        try: # Try to execute the following code block
            if hasattr(self.nn,'pr'): # If the model has a priority replay attribute
                self.nn.pr.TD=np.array(0) # Set it to zero array
        except Exception as e: # If an exception occurs
            raise e # Raise the exception and exit the method
        self.suspend=False # Set the suspend flag to False (not suspending training)
        self.save_epi=None # Set the save episode flag to None (not saving episodes)
        self.episode_set=[] # Initialize the episode set list to empty list (no episodes saved)
        self.state_pool=None # Set the state pool to None (no states stored)
        self.action_pool=None # Set the action pool to None (no actions stored)
        self.next_state_pool=None # Set the next state pool to None (no next states stored)
        self.reward_pool=None # Set the reward pool to None (no rewards stored)
        self.done_pool=None # Set the done pool to None (no done flags stored)
        self.reward_list=[] # Initialize the reward list to empty list (no rewards recorded)
        self.loss=0 # Set the loss value to zero (no loss calculated)
        self.loss_list=[] # Initialize the loss list to empty list (no losses recorded)
        self.sc=0 # Set the step counter to zero (no steps taken)
        self.total_episode=0 # Set the total episode counter to zero (no episodes completed)
        self.time=0 # Set the time value to zero (no time elapsed)
        self.total_time=0 # Set the total time value to zero (no total time elapsed)
        return # Return nothing and exit the method
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None): # This is a method for setting up some parameters
        if epsilon!=None: # If epsilon value is given
            self.epsilon=epsilon # Set it to the given value
        if episode_step!=None: # If episode step value is given
            self.episode_step=episode_step # Set it to the given value
        if pool_size!=None: # If pool size value is given
            self.pool_size=pool_size # Set it to the given value
        if batch!=None: # If batch size value is given
            self.batch=batch # Set it to the given value
        if update_step!=None: # If update step value is given
            self.update_step=update_step # Set it to the given value
        if trial_count!=None: # If trial count value is given
            self.trial_count=trial_count # Set it to the given value
        if criterion!=None: # If criterion value is given
            self.criterion=criterion # Set it to the given value
        self.action_vec() # Call the action vector method
        return
    
    
    def epsilon_greedy_policy(self,s): # This is a method for implementing epsilon-greedy policy
        action_prob=self.action_one*self.epsilon/len(self.action_one) # Create a vector of action probabilities with epsilon/number of actions for each action
        try:
            if hasattr(self.platform,'DType'): # If the platform is tensorflow
                best_a=np.argmax(self.nn.nn.fp(s)) # Get the best action by using the network's forward propagation method on the state
                action_prob[best_a]+=1-self.epsilon # Add 1-epsilon to the best action's probability
            else: # If the platform is pytorch
                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device) # Convert the state to a tensor with float type and send it to the device (cpu or gpu)
                best_a=self.nn.nn(s).argmax() # Get the best action by using the network's forward method on the state
                action_prob[best_a.numpy()]+=1-self.epsilon  # Add 1-epsilon to the best action's probability and convert it to numpy array
        except Exception as e: 
            raise e 
        return action_prob # Return the action probability vector
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch): # This is a method for optimizing the network parameters with a batch of data
        if hasattr(self.platform,'DType'): # If the platform is tensorflow
            loss=self.tf_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # Call the tensorflow optimization method with the data batch and get the loss value
        else: # If the platform is pytorch
            loss=self.pytorch_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # Call the pytorch optimization method with the data batch and get the loss value
        return loss # Return the loss value
    
    
    def opt_ol(self,state,action,next_state,reward,done): # This is a method for optimizing the network parameters with online data (one sample)
        if hasattr(self.platform,'DType'): # If the platform is tensorflow
            loss=self.tf_opt(state,action,next_state,reward,done) # Call the tensorflow optimization method with the online data and get the loss value
        else: # If the platform is pytorch
            loss=self.pytorch_opt(state,action,next_state,reward,done) # Call the pytorch optimization method with the online data and get the loss value
        return loss # Return the loss value
    
    
    def pool(self,s,a,next_s,r,done): # This is a method for storing data in the pool
        if type(self.state_pool)!=np.ndarray and self.state_pool==None: # If this is the first time to store data in the pool (the pool is None)
            self.state_pool=s # Set the state pool to be equal to s (the state)
            if type(a)==int: # If a (the action) is an integer (discrete action space)
                a=np.array(a) # Convert it to a numpy array
                self.action_pool=np.expand_dims(a,axis=0) # Expand its dimension by one and set it as the action pool 
            else: # If a (the action) is not an integer (continuous action space)
                self.action_pool=a # Set it as the action pool
            self.next_state_pool=np.expand_dims(next_s,axis=0) # Expand the dimension of next_s (the next state) by one and set it as the next state pool
            self.reward_pool=np.expand_dims(r,axis=0) # Expand the dimension of r (the reward) by one and set it as the reward pool
            self.done_pool=np.expand_dims(done,axis=0) # Expand the dimension of done (the done flag) by one and set it as the done pool
        else: # If this is not the first time to store data in the pool (the pool is not None)
            self.state_pool=np.concatenate((self.state_pool,s),0) # Concatenate s (the state) with the state pool along the first axis
            if type(a)==int: # If a (the action) is an integer (discrete action space)
                a=np.array(a) # Convert it to a numpy array
                self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0) # Expand its dimension by one and concatenate it with the action pool along the first axis
            else: # If a (the action) is not an integer (continuous action space)
                self.action_pool=np.concatenate((self.action_pool,a),0) # Concatenate a (the action) with the action pool along the first axis
            self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(next_s,axis=0)),0) # Expand the dimension of next_s (the next state) by one and concatenate it with the next state pool along the first axis
            self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0) # Expand the dimension of r (the reward) by one and concatenate it with the reward pool along the first axis
            self.done_pool=np.concatenate((self.done_pool,np.expand_dims(done,axis=0)),0) # Expand the dimension of done (the done flag) by one and concatenate it with the done pool along the first axis
        if len(self.state_pool)>self.pool_size: # If the length of the state pool exceeds the maximum size of the pool
            self.state_pool=self.state_pool[1:] # Remove the first element of the state pool
            self.action_pool=self.action_pool[1:] # Remove the first element of the action pool
            self.next_state_pool=self.next_state_pool[1:] # Remove the first element of the next state pool
            self.reward_pool=self.reward_pool[1:] # Remove the first element of the reward pool
            self.done_pool=self.done_pool[1:] # Remove the first element of the done pool
        return # Return nothing
    
    
    def choose_action(self,s): # This is a method for choosing an action based on a state
        if hasattr(self.nn,'nn'): # If the model has a network attribute
            if hasattr(self.platform,'DType'): # If the platform is tensorflow
                if self.epsilon==None: # If epsilon value is None
                    self.epsilon=self.nn.epsilon(self.sc) # Set it to be equal to the model's epsilon method with sc (step counter) as input
                action_prob=self.epsilon_greedy_policy(s) # Get the action probability vector by using epsilon-greedy policy method with s (state) as input
                a=np.random.choice(self.action_count,p=action_prob) # Choose an action randomly according to the action probability vector
            else: # If the platform is pytorch
                if self.epsilon==None: # If epsilon value is None
                    self.epsilon=self.nn.epsilon(self.sc) # Set it to be equal to the model's epsilon method with sc (step counter) as input
                action_prob=self.epsilon_greedy_policy(s) # Get the action probability vector by using epsilon-greedy policy method with s (state) as input
                a=np.random.choice(self.action_count,p=action_prob.numpy()) # Choose an action randomly according to the action probability vector converted to numpy array
        else: # If the model does not have a network attribute 
            if hasattr(self.nn,'action'): # If the model has an action attribute 
                if hasattr(self.platform,'DType'): # If the platform is tensorflow 
                    if self.epsilon==None:  # If epsilon value is None 
                        self.epsilon=self.nn.epsilon(self.sc)  # Set it to be equal to the model's epsilon method with sc (step counter) as input 
                    if hasattr(self.nn,'discriminator'):  # If the model has a discriminator attribute 
                        a=self.nn.action(s)  # Get an action by using model's action method with s (state) as input 
                        reward=self.nn.discriminator(s,a)  # Get a reward by using model's discriminator method with s (state) and a (action) as inputs 
                        s=np.squeeze(s)  # Squeeze s (state) to remove redundant
                    else: # If the model does not have a discriminator attribute 
                        a=self.nn.action(s).numpy() # Get an action by using model's action method with s (state) as input and convert it to numpy array
                else: # If the platform is pytorch 
                    if self.epsilon==None:  # If epsilon value is None 
                        self.epsilon=self.nn.epsilon(self.sc)  # Set it to be equal to the model's epsilon method with sc (step counter) as input 
                    if hasattr(self.nn,'discriminator'):  # If the model has a discriminator attribute 
                        a=self.nn.action(s)  # Get an action by using model's action method with s (state) as input 
                        reward=self.nn.discriminator(s,a)  # Get a reward by using model's discriminator method with s (state) and a (action) as inputs 
                        s=np.squeeze(s)  # Squeeze s (state) to remove redundant dimensions
                    else: # If the model does not have a discriminator attribute 
                        a=self.nn.action(s).detach().numpy() # Get an action by using model's action method with s (state) as input and convert it to numpy array
            else: # If the model does not have an action attribute 
                if hasattr(self.platform,'DType'): # If the platform is tensorflow 
                    a=(self.nn.actor.fp(s)+self.nn.noise()).numpy() # Get an action by using model's actor network's forward propagation method with s (state) and noise as inputs and convert it to numpy array
                else: # If the platform is pytorch 
                    s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device) # Convert the state to a tensor with float type and send it to the device (cpu or gpu)
                    a=(self.nn.actor(s)+self.nn.noise()).detach().numpy() # Get an action by using model's actor network's forward method with s (state) and noise as inputs and convert it to numpy array
        if hasattr(self.nn,'discriminator'): # If the model has a discriminator attribute 
            return a,reward # Return the action and the reward
        else: # If the model does not have a discriminator attribute 
            return a,None # Return the action and None
    
    
    def _train(self): # This is a method for training the network with data from the pool
        if len(self.state_pool)<self.batch: # If the length of the state pool is less than the batch size
            return np.array(0.) # Return zero array
        else: # If the length of the state pool is greater than or equal to the batch size
            loss=0 # Initialize loss value to zero
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch) # Calculate the number of batches by dividing the length of the state pool by the batch size and rounding down
            if len(self.state_pool)%self.batch!=0: # If there is a remainder after dividing the length of the state pool by the batch size
                batches+=1 # Add one more batch
            if hasattr(self.nn,'data_func'): # If the model has a data function attribute
                for j in range(batches): # For each batch
                    self.suspend_func() # Call the suspend function to check if training needs to be suspended
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch) # Get a batch of data by using model's data function with pools and batch size as inputs
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # Get a batch loss value by using opt method with data batch as inputs
                    loss+=batch_loss # Add batch loss value to total loss value
                    if hasattr(self.nn,'bc'): # If the model has a batch counter attribute
                        try:
                            self.nn.bc.assign_add(1) # Try to add one to the batch counter using assign_add method
                        except Exception: 
                            self.nn.bc+=1 # If failed, use normal addition instead
                if len(self.state_pool)%self.batch!=0:  # If there is a remainder after dividing the length of the state pool by the batch size
                    self.suspend_func()  # Call the suspend function to check if training needs to be suspended
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch)  # Get a batch of data by using model's data function with pools and batch size as inputs
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)  # Get a batch loss value by using opt method with data batch as inputs
                    loss+=batch_loss  # Add batch loss value to total loss value
                    if hasattr(self.nn,'bc'):  # If the model has a batch counter attribute
                        try:
                            self.nn.bc.assign_add(1)  # Try to add one to the batch counter using assign_add method
                        except Exception: 
                            self.nn.bc+=1  # If failed, use normal addition instead
            else: # If the model does not have a data function attribute
                j=0 # Initialize j to zero
                train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch) # Create a tensorflow dataset from the pools, shuffle it and batch it
                for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds: # For each batch of data from the dataset
                    self.suspend_func() # Call the suspend function to check if training needs to be suspended
                    if hasattr(self.platform,'DType'): # If the platform is tensorflow 
                        pass # Do nothing
                    else: # If the platform is pytorch 
                        state_batch=state_batch.numpy() # Convert state batch to numpy array
                        action_batch=action_batch.numpy() # Convert action batch to numpy array
                        next_state_batch=next_state_batch.numpy() # Convert next state batch to numpy array
                        reward_batch=reward_batch.numpy() # Convert reward batch to numpy array
                        done_batch=done_batch.numpy() # Convert done batch to numpy array
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch) # Get a batch loss value by using opt method with data batch as inputs
                    loss+=batch_loss # Add batch loss value to total loss value
                    j+=1 # Increase j by one
                    if hasattr(self.nn,'bc'): # If the model has a batch counter attribute 
                        try:
                            self.nn.bc.assign_add(1) # Try to add one to the batch counter using assign_add method
                        except Exception: 
                            self.nn.bc+=1 # If failed, use normal addition instead
            if self.update_step!=None: # If update step value is not None 
                if self.sc%self.update_step==0: # If the step counter is divisible by the update step value 
                    self.nn.update_param() # Call the model's update parameter method 
            else: # If update step value is None 
                self.nn.update_param() # Call the model's update parameter method 
        if hasattr(self.platform,'DType'): # If the platform is tensorflow 
            loss=loss.numpy()/batches # Convert loss value to numpy array and divide it by batches 
        else: # If the platform is pytorch 
            loss=loss.detach().numpy()/batches # Detach loss value from computation graph and convert it to numpy array and divide it by batches 
        return loss # Return loss value
    
    
    def train_(self):  # This is a method for training the network for one episode
        episode=[]  # Initialize episode list to empty list
        self.reward=0  # Initialize reward value to zero
        s=self.nn.env(initial=True)  # Get an initial state from the model's environment method with initial flag set to True
        if hasattr(self.platform,'DType'):  # If the platform is tensorflow 
            if type(self.nn.param[0])!=list:  # If the first element of model's parameter list is not a list (single network)
                s=np.array(s,self.nn.param[0].dtype.name)  # Convert s (state) to a numpy array with the same data type as the first element of model's parameter list
            else:  # If the first element of model's parameter list is a list (multiple networks)
                s=np.array(s,self.nn.param[0][0].dtype.name)  # Convert s (state) to a numpy array with the same data type as the first element of the first element of model's parameter list
        if self.episode_step==None:  # If episode step value is None (no limit on steps per episode)
            while True:  # Loop indefinitely until done flag is True or break statement is executed
                s=np.expand_dims(s,axis=0)  # Expand s (state) dimension by one 
                a,reward=self.choose_action(s)  # Get an action and a reward by using choose action method with s (state) as input
                next_s,r,done=self.nn.env(a)  # Get next state, reward
                if hasattr(self.platform,'DType'):  # If the platform is tensorflow 
                    if type(self.nn.param[0])!=list:  # If the first element of model's parameter list is not a list (single network)
                        next_s=np.array(next_s,self.nn.param[0].dtype.name)  # Convert next_s (next state) to a numpy array with the same data type as the first element of model's parameter list
                        r=np.array(r,self.nn.param[0].dtype.name)  # Convert r (reward) to a numpy array with the same data type as the first element of model's parameter list
                        done=np.array(done,self.nn.param[0].dtype.name)  # Convert done (done flag) to a numpy array with the same data type as the first element of model's parameter list
                    else:  # If the first element of model's parameter list is a list (multiple networks)
                        next_s=np.array(next_s,self.nn.param[0][0].dtype.name)  # Convert next_s (next state) to a numpy array with the same data type as the first element of the first element of model's parameter list
                        r=np.array(r,self.nn.param[0][0].dtype.name)  # Convert r (reward) to a numpy array with the same data type as the first element of the first element of model's parameter list
                        done=np.array(done,self.nn.param[0][0].dtype.name)  # Convert done (done flag) to a numpy array with the same data type as the first element of the first element of model's parameter list
                if hasattr(self.nn,'pool'):  # If the model has a pool attribute 
                    if hasattr(self.nn,'discriminator'):  # If the model has a discriminator attribute 
                        self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,reward,done])  # Call the model's pool method with pools and data as inputs
                    else:  # If the model does not have a discriminator attribute 
                        self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,r,done])  # Call the model's pool method with pools and data as inputs
                else:  # If the model does not have a pool attribute 
                    self.pool(s,a,next_s,r,done)  # Call the pool method with data as inputs
                if hasattr(self.nn,'pr'):  # If the model has a priority replay attribute 
                    self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD)  # Append the initial TD value to the TD array of priority replay
                    if len(self.state_pool)>self.pool_size:  # If the length of the state pool exceeds the maximum size of the pool
                        TD=np.array(0)  # Create a zero array for TD value
                        self.nn.pr.TD=np.append(TD,self.nn.pr.TD[2:])  # Append TD value to the TD array of priority replay and remove the first two elements
                self.reward=r+self.reward  # Add r (reward) to total reward value
                loss=self._train()  # Call the _train method and get loss value
                self.sc+=1  # Increase sc (step counter) by one
                if done:  # If done flag is True 
                    if self.save_episode==True:  # If save episode flag is True 
                        episode=[s,a,next_s,r]  # Set episode list to be equal to data
                    self.reward_list.append(self.reward)  # Append total reward value to reward list
                    return loss,episode,done  # Return loss value, episode list, and done flag
                elif self.save_episode==True:  # If save episode flag is True and done flag is False 
                    episode=[s,a,next_s,r]  # Set episode list to be equal to data
                s=next_s  # Set s (state) to be equal to next_s (next state)
        else:  # If episode step value is not None (there is a limit on steps per episode)
            for _ in range(self.episode_step):  # For each step in episode step value 
                s=np.expand_dims(s,axis=0)  # Expand s (state) dimension by one 
                a,reward=self.choose_action(s)  # Get an action and a reward by using choose action method with s (state) as input
                next_s,r,done=self.nn.env(a)  # Get next state, reward, and done flag by using model's environment method with a (action) as input
                if hasattr(self.platform,'DType'):  # If the platform is tensorflow 
                    if type(self.nn.param[0])!=list:  # If the first element of model's parameter list is not a list (single network)
                        next_s=np.array(next_s,self.nn.param[0].dtype.name)  # Convert next_s (next state) to a numpy array with the same data type as the first element of model's parameter list
                        r=np.array(r,self.nn.param[0].dtype.name)  # Convert r (reward) to a numpy array with the same data type as the first element of model's parameter list
                        done=np.array(done,self.nn.param[0].dtype.name)  # Convert done (done flag) to a numpy array with the same data type as the first element of model's parameter list
                    else:  # If the first element of model's parameter list is a list (multiple networks)
                        next_s=np.array(next_s,self.nn.param[0][0].dtype.name)  # Convert next_s (next state) to a numpy array with the same data type as the first element of the first element of model's parameter list
                        r=np.array(r,self.nn.param[0][0].dtype.name)  # Convert r (reward) to a numpy array with the same data type as the first element of the first element of model's parameter list
                        done=np.array(done,self.nn.param[0][0].dtype.name)  # Convert done (done flag) to a numpy array with the same data type as the first element of the first element of model's parameter list
                if hasattr(self.nn,'pool'):  # If the model has a pool attribute 
                    if hasattr(self.nn,'discriminator'):  # If the model has a discriminator attribute 
                        self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,reward,done])  # Call the model's pool method with pools and data as inputs
                    else:  # If the model does not have a discriminator attribute 
                        self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,r,done])  # Call the model's pool method with pools and data as inputs
                else:  # If the model does not have a pool attribute 
                    self.pool(s,a,next_s,r,done)  # Call the pool method with data as inputs
                if hasattr(self.nn,'pr'):  # If the model has a priority replay attribute 
                    self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD)  # Append the initial TD value to the TD array of priority replay
                    if len(self.state_pool)>self.pool_size:  # If the length of the state pool exceeds the maximum size of the pool
                        TD=np.array(0)  # Create a zero array for TD value
                        self.nn.pr.TD=np.append(TD,self.nn.pr.TD[2:])  # Append TD value to the TD array of priority replay and remove the first two elements
                self.reward=r+self.reward  # Add r (reward) to total reward value
                loss=self._train()  # Call the _train method and get loss value
                self.sc+=1  # Increase sc (step counter) by one
                if done:  # If done flag is True 
                    if self.save_episode==True:  # If save episode flag is True 
                        episode=[s,a,next_s,r]  # Set episode list to be equal to data
                    self.reward_list.append(self.reward)  # Append total reward value to reward list
                    return loss,episode,done  # Return loss value, episode list, and done flag
                elif self.save_episode==True:  # If save episode flag is True and done flag is False 
                    episode=[s,a,next_s,r]  # Set episode list to be equal to data
                s=next_s  # Set s (state) to be equal to next_s (next state)
        self.reward_list.append(self.reward)  # Append total reward value to reward list
        return loss,episode,done  # Return loss value, episode list, and done flag
    
    
    def train(self,episode_count,save=None,one=True,p=None,s=None):  # This is a method for training the network for multiple episodes
        avg_reward=None  # Initialize average reward value to None
        if p==None:  # If p value is None (p is used for printing frequency)
            self.p=9   # Set it to be 9 (print every 10 episodes)
        else:  # If p value is given 
            self.p=p-1  # Set it to be p minus one (print every p episodes)
        if s==None:  # If s value is None (s is used for saving frequency)
            self.s=1  # Set it to be 1 (save every episode)
            self.file_list=None  # Set file list to be None (no need to store file names)
        else:  # If s value is given 
            self.s=s-1  # Set it to be s minus one (save every s episodes)
            self.file_list=[]  # Initialize file list to empty list (need to store file names)
        if episode_count!=None:  # If episode count value is not None (there is a limit on episodes)
            for i in range(episode_count):  # For each episode in episode count value 
                t1=time.time()  # Record the start time of the episode
                loss,episode,done=self.train_()  # Call the train_ method and get loss value, episode list, and done flag
                if self.trial_count!=None:  # If trial count value is not None (need to calculate average reward)
                    if len(self.reward_list)>=self.trial_count:  # If the length of the reward list is greater than or equal to the trial count value 
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])  # Calculate the average reward by using the mean function on the last trial count elements of the reward list
                        if self.criterion!=None and avg_reward>=self.criterion:  # If criterion value is not None and average reward is greater than or equal to criterion value (achieved the goal)
                            t2=time.time()  # Record the end time of the episode
                            self.total_time+=(t2-t1)  # Add the episode time to the total time
                            time_=self.total_time-int(self.total_time)  # Get the decimal part of the total time
                            if time_<0.5:  # If the decimal part is less than 0.5
                                self.total_time=int(self.total_time)  # Round down the total time
                            else:  # If the decimal part is greater than or equal to 0.5
                                self.total_time=int(self.total_time)+1  # Round up the total time
                            print('episode:{0}'.format(self.total_episode))  # Print the current episode number
                            print('last loss:{0:.6f}'.format(loss))  # Print the last loss value with six decimal places
                            print('average reward:{0}'.format(avg_reward))  # Print the average reward value
                            print()  
                            print('time:{0}s'.format(self.total_time))  # Print the total time in seconds
                            return  # Return nothing and exit the method
                self.loss=loss  # Set loss attribute to be equal to loss value
                self.loss_list.append(loss)  # Append loss value to loss list
                self.total_episode+=1  # Increase total episode by one
                if episode_count%10!=0:  # If episode count is not divisible by 10 
                    p=episode_count-episode_count%self.p  # Calculate p by subtracting the remainder of dividing episode count by p from episode count 
                    p=int(p/self.p)  # Divide p by p and round down 
                    s=episode_count-episode_count%self.s   # Calculate s by subtracting the remainder of dividing episode count by s from episode count 
                    s=int(s/self.s)   # Divide s by s and round down 
                else:   # If episode count is divisible by 10 
                    p=episode_count/(self.p+1)   # Divide episode count by p plus one and round down 
                    p=int(p)   # Convert p to integer 
                    s=episode_count/(self.s+1)   # Divide episode count by s plus one and round down 
                    s=int(s)   # Convert s to integer 
                if p==0:   # If p is zero (should not happen)
                    p=1   # Set it to be one (print every episode)
                if s==0:   # If s is zero (should not happen)
                    s=1   # Set it to be one (save every episode)
                if i%p==0:   # If i (current episode number) is divisible by p (print frequency)
                    if len(self.state_pool)>=self.batch:   # If the length of the state pool is greater than or equal to the batch size (enough data for training)
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))   # Print the current episode number and loss value with six decimal places
                    if avg_reward!=None:   # If average reward value is not None (calculated average reward)
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))   # Print the current episode number and average reward value
                    else:   # If average reward value is None (did not calculate average reward)
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))   # Print the current episode number and total reward value
                    print()  
                if save!=None and i%s==0:   # If save value is not None (need to save data) and i (current episode number) is divisible by s (save frequency)
                    self.save(self.total_episode,one)   # Call the save method with total episode and one flag as inputs
                if self.save_episode==True:   # If save episode flag is True (need to save episodes)
                    if done:   # If done flag is True (episode ended)
                        episode.append('done')   # Append 'done' to episode list
                    self.episode_set.append(episode)   # Append episode list to episode set list
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count:  # If max episode count value is not None (there is a limit on episodes to save) and the length of episode set list is greater than or equal to max episode count value 
                        self.save_episode=False  # Set save episode flag to False (stop saving episodes)
                try:  # Try to execute the following code block
                    self.nn.ec.assign_add(1)  # Add one to the model's episode counter using assign_add method
                except Exception:  # If an exception occurs
                    pass  # Do nothing and ignore the exception
                t2=time.time()  # Record the end time of the episode
                self.time+=(t2-t1)  # Add the episode time to the time attribute
        else:  # If episode count value is None (no limit on episodes)
            i=0  # Initialize i to zero
            while True:  # Loop indefinitely until break statement is executed
                t1=time.time()  # Record the start time of the episode
                loss,episode,done=self.train_()  # Call the train_ method and get loss value, episode list, and done flag
                if self.trial_count!=None:  # If trial count value is not None (need to calculate average reward)
                    if len(self.reward_list)==self.trial_count:  # If the length of the reward list is equal to the trial count value 
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])  # Calculate the average reward by using the mean function on the last trial count elements of the reward list
                        if self.criterion!=None and avg_reward>=self.criterion:  # If criterion value is not None and average reward is greater than or equal to criterion value (achieved the goal)
                            t2=time.time()  # Record the end time of the episode
                            self.total_time+=(t2-t1)  # Add the episode time to the total time
                            time_=self.total_time-int(self.total_time)  # Get the decimal part of the total time
                            if time_<0.5:  # If the decimal part is less than 0.5
                                self.total_time=int(self.total_time)  # Round down the total time
                            else:  # If the decimal part is greater than or equal to 0.5
                                self.total_time=int(self.total_time)+1  # Round up the total time
                            print('episode:{0}'.format(self.total_episode))  # Print the current episode number
                            print('last loss:{0:.6f}'.format(loss))  # Print the last loss value with six decimal places
                            print('average reward:{0}'.format(avg_reward))  # Print the average reward value
                            print()  
                            print('time:{0}s'.format(self.total_time))  # Print the total time in seconds
                            return  # Return nothing and exit the method
                self.loss=loss  # Set loss attribute to be equal to loss value
                self.loss_list.append(loss)  # Append loss value to loss list
                i+=1  # Increase i by one 
                self.total_episode+=1  # Increase total episode by one 
                if episode_count%10!=0:  # If episode count is not divisible by 10 
                    p=episode_count-episode_count%self.p  # Calculate p by subtracting the remainder of dividing episode count by p from episode count 
                    p=int(p/self.p)  # Divide p by p and round down 
                    s=episode_count-episode_count%self.s   # Calculate s by subtracting the remainder of dividing episode count by s from episode count 
                    s=int(s/self.s)   # Divide s by s and round down 
                else:   # If episode count is divisible by 10 
                    p=episode_count/(self.p+1)   # Divide episode count by p plus one and round down 
                    p=int(p)   # Convert p to integer 
                    s=episode_count/(self.s+1)   # Divide episode count by s plus one and round down 
                    s=int(s)   # Convert s to integer 
                if i%p==0:   # If i (current episode number) is divisible by p (print frequency)
                    if len(self.state_pool)>=self.batch:   # If the length of the state pool is greater than or equal to the batch size (enough data for training)
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))   # Print the current episode number and loss value with six decimal places
                    if avg_reward!=None:   # If average reward value is not None (calculated average reward)
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))   # Print the current episode number and average reward value
                    else:   # If average reward value is None (did not calculate average reward)
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))   # Print the current episode number and total reward value
                    print()  
                if save!=None and i%s==0:   # If save value is not None (need to save data) and i (current episode number) is divisible by s (save frequency)
                    self.save(self.total_episode,one)   # Call the save method with total episode and one flag as inputs
                if self.save_episode==True:   # If save episode flag is True (need to save episodes)
                    if done:   # If done flag is True (episode ended)
                        episode.append('done')   # Append 'done' to episode list
                    self.episode_set.append(episode)   # Append episode list to episode set list
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count:  # If max episode count value is not None (there is a limit on episodes to save) and the length of episode set list is greater than or equal to max episode count value 
                        self.save_episode=False  # Set save episode flag to False (stop saving episodes)
                try:  # Try to execute the following code block
                    self.nn.ec.assign_add(1)  # Add one to the model's episode counter using assign_add method
                except Exception:  # If an exception occurs
                    self.nn.ec+=1  # Use normal addition instead
                t2=time.time()  # Record the end time of the episode
                self.time+=(t2-t1)  # Add the episode time to the time attribute
        time_=self.time-int(self.time)  # Get the decimal part of the time attribute
        if time_<0.5:  # If the decimal part is less than 0.5
            self.total_time=int(self.time)  # Round down the time attribute
        else:  # If the decimal part is greater than or equal to 0.5
            self.total_time=int(self.time)+1  # Round up the time attribute
        self.total_time+=self.time  # Add the time attribute to the total time attribute
        print('last loss:{0:.6f}'.format(loss))  # Print the last loss value with six decimal places
        print('last reward:{0}'.format(self.reward))  # Print the last reward value
        print()  
        print('time:{0}s'.format(self.time))  # Print the time attribute in seconds
        return  # Return nothing
    
    
    def train_online(self):  # This is a method for training the network online (without using a pool)
        while True:  # Loop indefinitely until break statement is executed
            if hasattr(self.nn,'save'):  # If the model has a save attribute 
                self.nn.save(self.save)  # Call the model's save method with save attribute as input
            if hasattr(self.nn,'stop_flag'):  # If the model has a stop flag attribute 
                if self.nn.stop_flag==True:  # If the stop flag is True (need to stop training)
                    return  # Return nothing and exit the method
            if hasattr(self.nn,'stop_func'):  # If the model has a stop function attribute 
                if self.nn.stop_func():  # If the stop function returns True (need to stop training)
                    return  # Return nothing and exit the method
            if hasattr(self.nn,'suspend_func'):  # If the model has a suspend function attribute 
                self.nn.suspend_func()  # Call the suspend function to check if training needs to be suspended
            data=self.nn.online()  # Get online data by using model's online method
            if data=='stop':  # If data is 'stop' (need to stop training)
                return  # Return nothing and exit the method
            elif data=='suspend':  # If data is 'suspend' (need to suspend training)
                self.nn.suspend_func()  # Call the suspend function to suspend training
            loss=self.opt_ol(data[0],data[1],data[2],data[3],data[4])  # Get loss value by using opt_ol method with data as inputs
            loss=loss.numpy()  # Convert loss value to numpy array
            self.nn.train_loss_list.append(loss)  # Append loss value to model's train loss list
            if len(self.nn.train_acc_list)==self.nn.max_length:  # If the length of model's train accuracy list is equal to model's max length value 
                del self.nn.train_acc_list[0]  # Delete the first element of model's train accuracy list
            if hasattr(self.nn,'counter'):  # If the model has a counter attribute 
                self.nn.counter+=1  # Increase the counter by one
        return  # Return nothing
