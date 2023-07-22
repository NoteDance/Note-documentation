# import the necessary modules
from tensorflow import function
from tensorflow import data as tf_data
import numpy as np
import statistics
import time

# define a class named kernel
class kernel:
    # define the initialization method
    def __init__(self,nn=None,save_episode=False):
        # assign the nn parameter to the self.nn attribute
        self.nn=nn
        # if the nn attribute has a km attribute, assign 1 to it
        if hasattr(self.nn,'km'):
            self.nn.km=1
        # initialize the platform attribute as None
        self.platform=None
        # initialize the state_pool, action_pool, next_state_pool, reward_pool, done_pool attributes as None
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        # initialize the episode_set attribute as an empty list
        self.episode_set=[]
        # initialize the epsilon attribute as None
        self.epsilon=None
        # initialize the episode_step attribute as None
        self.episode_step=None
        # initialize the pool_size attribute as None
        self.pool_size=None
        # initialize the batch attribute as None
        self.batch=None
        # initialize the update_step attribute as None
        self.update_step=None
        # initialize the trial_count attribute as None
        self.trial_count=None
        # initialize the criterion attribute as None
        self.criterion=None
        # initialize the reward_list attribute as an empty list
        self.reward_list=[]
        # initialize the suspend attribute as False
        self.suspend=False
        # initialize the save_epi attribute as None
        self.save_epi=None
        # initialize the max_episode_count attribute as None
        self.max_episode_count=None
        # assign the save_episode parameter to the save_episode attribute 
        self.save_episode=save_episode 
        # assign the filename parameter to the filename attribute
        self.filename='save.dat'
        # initialize the loss attribute as 0
        self.loss=0
        # initialize the loss_list attribute as an empty list
        self.loss_list=[]
        # initialize the sc attribute as 0
        self.sc=0
        # initialize the total_episode attribute as 0
        self.total_episode=0
        # initialize the time attribute as 0
        self.time=0
        # initialize the total_time attribute as 0
        self.total_time=0
        
        # define a method named action_vec
        def action_vec(self):
            # if the epsilon attribute is not None, assign a numpy array of ones with the length of action_count and multiply it by epsilon divided by the length of action_one to the action_one attribute
            if self.epsilon!=None:
                self.action_one=np.ones(self.action_count,dtype=np.int8)
            # return nothing
            return
        
        # define a method named init
        def init(self):
            # try to execute the following statements
            try:
                # if the nn attribute has a pr attribute, assign a numpy array of 0 to its TD attribute
                if hasattr(self.nn,'pr'):
                    self.nn.pr.TD=np.array(0)
            # if an exception occurs, raise it
            except Exception as e:
                raise e
            # assign False to the suspend attribute
            self.suspend=False
            # assign None to the save_epi attribute
            self.save_epi=None
            # assign an empty list to the episode_set attribute
            self.episode_set=[]
            # assign None to the state_pool, action_pool, next_state_pool, reward_pool, done_pool attributes 
            self.state_pool=None
            self.action_pool=None
            self.next_state_pool=None
            self.reward_pool=None
            self.done_pool=None
            # assign an empty list to the reward_list attribute 
            self.reward_list=[]
            # assign 0 to the loss attribute 
            self.loss=0
            # assign an empty list to the loss_list attribute 
            self.loss_list=[]
            # assign 0 to the sc attribute 
            self.sc=0
            # assign 0 to the total_episode attribute 
            self.total_episode=0
            # assign 0 to the time attribute 
            self.time=0
            # assign 0 to the total_time attribute 
            self.total_time=0
        
        # define a method named set_up
        def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_count=None,criterion=None):
            # if epsilon is not None, assign it to the epsilon attribute
            if epsilon!=None:
                self.epsilon=epsilon
            # if episode_step is not None, assign it to the episode_step attribute
            if episode_step!=None:
                self.episode_step=episode_step
            # if pool_size is not None, assign it to the pool_size attribute
            if pool_size!=None:
                self.pool_size=pool_size
            # if batch is not None, assign it to the batch attribute
            if batch!=None:
                self.batch=batch
            # if update_step is not None, assign it to the update_step attribute
            if update_step!=None:
                self.update_step=update_step
            # if trial_count is not None, assign it to the trial_count attribute
            if trial_count!=None:
                self.trial_count=trial_count
            # if criterion is not None, assign it to the criterion attribute
            if criterion!=None:
                self.criterion=criterion
            # call the action_vec method
            self.action_vec()
            # return nothing
            return
        
        # define a method named epsilon_greedy_policy
        def epsilon_greedy_policy(self,s):
            # assign a numpy array of ones with the length of action_count and multiply it by epsilon divided by the length of action_one to the action_prob variable
            action_prob=self.action_one*self.epsilon/len(self.action_one)
            # try to execute the following statements
            try:
                # assign the index of the maximum value of the output of the nn attribute's fp method with s as input to the best_a variable
                best_a=np.argmax(self.nn.nn.fp(s))
                # add 1 minus epsilon to the best_a element of action_prob
                action_prob[best_a]+=1-self.epsilon
            # if an exception occurs, execute the following statements
            except Exception as e:
                # if the platform attribute has a DType attribute, raise the exception
                if hasattr(self.platform,'DType'):
                    raise e
                # else, assign the index of the maximum value of the output of the nn attribute's nn method with s as input to the best_a variable
                else:
                    best_a=self.nn.nn(s).argmax()
                    # add 1 minus epsilon to the best_a element of action_prob and convert it to numpy array
                    action_prob[best_a.numpy()]+=1-self.epsilon
            # return action_prob
            return action_prob
        
        # define a method named get_reward
        def get_reward(self,max_step=None,seed=None):
            # initialize the reward variable as 0
            reward=0
            # if seed is None, assign the output of the genv attribute's reset method to the s variable
            if seed==None:
                s=self.genv.reset()
            # else, assign the output of the genv attribute's reset method with seed as input to the s variable
            else:
                s=self.genv.reset(seed=seed)
            # if max_step is not None, execute the following statements
            if max_step!=None:
                # use a for loop to iterate from 0 to max_step
                for i in range(max_step):
                    # if the end_flag attribute is True, break the loop
                    if self.end_flag==True:
                        break
                    # if the nn attribute has a nn attribute, execute the following statements
                    if hasattr(self.nn,'nn'):
                        # if the platform attribute has a DType attribute, execute the following statements
                        if hasattr(self.platform,'DType'):
                            # expand the dimension of s along axis 0 and assign it back to s
                            s=np.expand_dims(s,axis=0)
                            # assign the index of the maximum value of the output of the nn attribute's nn method with s as input to the a variable
                            a=np.argmax(self.nn.nn.fp(s))
                        # else, execute the following statements
                        else:
                            # expand the dimension of s along axis 0 and assign it back to s
                            s=np.expand_dims(s,axis=0)
                            # convert s to a tensor with float dtype and assign it to the device attribute of the nn attribute and assign it back to s
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            # assign the index of the maximum value of the output of the nn attribute's nn method with s as input to the a variable and convert it to numpy array
                            a=self.nn.nn(s).detach().numpy().argmax()
                    # else, execute the following statements
                    else:
                        # if the nn attribute has an action attribute, execute the following statements
                        if hasattr(self.nn,'action'):
                            # if the platform attribute has a DType attribute, execute the following statements
                            if hasattr(self.platform,'DType'):
                                # expand the dimension of s along axis 0 and assign it back to s
                                s=np.expand_dims(s,axis=0)
                                # assign the output of the nn attribute's action method with s as input to the a variable and convert it to numpy array
                                a=self.nn.action(s).numpy()
                            # else, execute the following statements
                            else:
                                # expand the dimension of s along axis 0 and assign it back to s
                                s=np.expand_dims(s,axis=0)
                                # convert s to a tensor with float dtype and assign it to the device attribute of the nn attribute and assign it back to s
                                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                                # assign the output of the nn attribute's action method with s as input to the a variable and convert it to numpy array and detach it from computation graph
                                a=self.nn.action(s).detach().numpy()
                        # else, execute the following statements
                        else:
                            # if the platform attribute has a DType attribute, execute the following statements
                            if hasattr(self.platform,'DType'):
                                # expand the dimension of s along axis 0 and assign it back to s
                                s=np.expand_dims(s,axis=0)
                                # assign the output of adding up output of fp method from actor attribute of nn with noise from nn as input to a variable and convert it to numpy array 
                                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                                # squeeze out any singleton dimensions from a and assign it back to a 
                                a=np.squeeze(a)
                            # else, execute the following statements 
                            else:
                                # expand the dimension of s along axis 0 and assign it back to s 
                                s=np.expand_dims(s,axis=0)
                                # convert s to a tensor with float dtype and assign it to device attribute of nn and assign it back to s 
                                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                                # assign output of adding up output from actor method from nn with noise from nn as input to a variable and convert it to numpy array and detach it from computation graph 
                                a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()
                                # squeeze out any singleton dimensions from a and assign it back to a 
                                a=np.squeeze(a)
                    # assign the output of the genv attribute's step method with a as input to the next_s, r, done variables
                    next_s,r,done,_=self.genv.step(a)
                    # assign next_s to s
                    s=next_s
                    # add r to reward and assign it back to reward
                    reward+=r
                    # if the nn attribute has a stop attribute, execute the following statements
                    if hasattr(self.nn,'stop'):
                        # if the output of the nn attribute's stop method with next_s as input is True, break the loop
                        if self.nn.stop(next_s):
                            break
                    # if done is True, break the loop
                    if done:
                        break
                # return reward
                return reward
            # else, execute the following statements
            else:
                # use a while loop to iterate indefinitely
                while True:
                    # if the end_flag attribute is True, break the loop
                    if self.end_flag==True:
                        break
                    # if the nn attribute has a nn attribute, execute the following statements
                    if hasattr(self.nn,'nn'):
                        # if the platform attribute has a DType attribute, execute the following statements
                        if hasattr(self.platform,'DType'):
                            # expand the dimension of s along axis 0 and assign it back to s
                            s=np.expand_dims(s,axis=0)
                            # assign the index of the maximum value of the output of the nn attribute's nn method with s as input to the a variable
                            a=np.argmax(self.nn.nn.fp(s))
                        # else, execute the following statements
                        else:
                            # expand the dimension of s along axis 0 and assign it back to s
                            s=np.expand_dims(s,axis=0)
                            # convert s to a tensor with float dtype and assign it to the device attribute of the nn attribute and assign it back to s
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            # assign the index of the maximum value of the output of the nn attribute's nn method with s as input to the a variable and convert it to numpy array
                            a=self.nn.nn(s).detach().numpy().argmax()
                    # else, execute the following statements
                    else:
                        # if the nn attribute has an action attribute, execute the following statements
                        if hasattr(self.nn,'action'):
                            # if the platform attribute has a DType attribute, execute the following statements
                            if hasattr(self.platform,'DType'):
                                # expand the dimension of s along axis 0 and assign it back to s
                                s=np.expand_dims(s,axis=0)
                                # assign the output of the nn attribute's action method with s as input to the a variable and convert it to numpy array 
                                a=self.nn.action(s).numpy()
                            # else, execute the following statements 
                            else:
                                # expand the dimension of s along axis 0 and assign it back to s 
                                s=np.expand_dims(s,axis=0)
                                # convert s to a tensor with float dtype and assign it to device attribute of nn and assign it back to s 
                                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                                # assign output of action method from nn with s as input to a variable and convert it to numpy array and detach it from computation graph 
                                a=self.nn.action(s).detach().numpy()
                        # else, execute the following statements 
                        else:
                            # if platform has DType attribute, execute following statements 
                            if hasattr(self.platform,'DType'):
                                # expand dimension of s along axis 0 and assign it back to s 
                                s=np.expand_dims(s,axis=0)
                                # assign output of adding up output from fp method from actor attribute of nn with noise from nn as input to a variable and convert it to numpy array 
                                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                                # squeeze out any singleton dimensions from a and assign it back to a 
                                a=np.squeeze(a)
                            # else, execute following statements 
                            else:
                                # expand dimension of s along axis 0 and assign it back to s 
                                s=np.expand_dims(s,axis=0)
                                # convert s to tensor with float dtype and assign it to device attribute of nn and assign it back to s 
                                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                                # assign output of adding up output from actor method from nn with noise from nn as input to a variable and convert it to numpy array and detach it from computation graph 
                                a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()
                                # squeeze out any singleton dimensions from a and assign it back to a 
                                a=np.squeeze(a)
                    # assign output of genv attribute's step method with a as input to next_s, r, done variables 
                    next_s,r,done,_=self.genv.step(a)
                    # assign next_s to s
                    s=next_s
                    # add r to reward and assign it back to reward
                    reward+=r
                    # if the nn attribute has a stop attribute, execute the following statements
                    if hasattr(self.nn,'stop'):
                        # if the output of the nn attribute's stop method with next_s as input is True, break the loop
                        if self.nn.stop(next_s):
                            break
                    # if done is True, break the loop
                    if done:
                        break
                # return reward
                return reward
        
        # define a method named get_episode
        def get_episode(self,max_step=None,seed=None):
            # initialize the counter variable as 0
            counter=0
            # initialize the episode variable as an empty list
            episode=[]
            # if seed is None, assign the output of the genv attribute's reset method to the s variable
            if seed==None:
                s=self.genv.reset()
            # else, assign the output of the genv attribute's reset method with seed as input to the s variable
            else:
                s=self.genv.reset(seed=seed)
            # initialize the end_flag attribute as False
            self.end_flag=False
            # use a while loop to iterate indefinitely
            while True:
                # if the end_flag attribute is True, break the loop
                if self.end_flag==True:
                    break
                # if the nn attribute has a nn attribute, execute the following statements
                if hasattr(self.nn,'nn'):
                    # if the platform attribute has a DType attribute, execute the following statements
                    if hasattr(self.platform,'DType'):
                        # expand the dimension of s along axis 0 and assign it back to s
                        s=np.expand_dims(s,axis=0)
                        # assign the index of the maximum value of the output of the nn attribute's nn method with s as input to the a variable
                        a=np.argmax(self.nn.nn.fp(s))
                    # else, execute the following statements 
                    else:
                        # expand the dimension of s along axis 0 and assign it back to s 
                        s=np.expand_dims(s,axis=0)
                        # convert s to a tensor with float dtype and assign it to the device attribute of the nn attribute and assign it back to s 
                        s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                        # assign the index of the maximum value of the output of the nn attribute's nn method with s as input to the a variable and convert it to numpy array 
                        a=self.nn.nn(s).detach().numpy().argmax()
                # else, execute the following statements 
                else:
                    # if the nn attribute has an action attribute, execute the following statements 
                    if hasattr(self.nn,'action'):
                        # if the platform attribute has a DType attribute, execute the following statements 
                        if hasattr(self.platform,'DType'):
                            # expand the dimension of s along axis 0 and assign it back to s 
                            s=np.expand_dims(s,axis=0)
                            # assign output of action method from nn with s as input to a variable and convert it to numpy array 
                            a=self.nn.action(s).numpy()
                        # else, execute following statements 
                        else:
                            # expand dimension of s along axis 0 and assign it back to s 
                            s=np.expand_dims(s,axis=0)
                            # convert s to tensor with float dtype and assign it to device attribute of nn and assign it back to s 
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            # assign output of action method from nn with s as input to a variable and convert it to numpy array and detach it from computation graph 
                            a=self.nn.action(s).detach().numpy()
                    # else, execute following statements 
                    else:
                        # if platform has DType attribute, execute following statements 
                        if hasattr(self.platform,'DType'):
                            # expand dimension of s along axis 0 and assign it back to s 
                            s=np.expand_dims(s,axis=0)
                            # assign output of adding up output from fp method from actor attribute of nn with noise from nn as input to a variable and convert it to numpy array 
                            a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
                            # squeeze out any singleton dimensions from a and assign it back to a 
                            a=np.squeeze(a)
                        # else, execute following statements 
                        else:
                            # expand dimension of s along axis 0 and assign it back to s 
                            s=np.expand_dims(s,axis=0)
                            # convert s to tensor with float dtype and assign it to device attribute of nn and assign it back to s 
                            s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                            # assign output of adding up output from actor method from nn with noise from nn as input to a variable and convert it to numpy array and detach it from computation graph 
                            a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()
                            # squeeze out any singleton dimensions from a and assign it back to a 
                            a=np.squeeze(a)
                # assign the output of the genv attribute's step method with a as input to the next_s, r, done variables
                next_s,r,done,_=self.genv.step(a)
                # if the end_flag attribute is True, break the loop
                if self.end_flag==True:
                    break
                # if done is True, execute the following statements
                if done:
                    # append a list of s, a, next_s, r to the episode variable
                    episode.append([s,a,next_s,r])
                    # append the string 'done' to the episode variable
                    episode.append('done')
                    # break the loop
                    break
                # else, execute the following statements
                else:
                    # append a list of s, a, next_s, r to the episode variable
                    episode.append([s,a,next_s,r])
                # if max_step is not None and counter equals max_step minus 1, break the loop
                if max_step!=None and counter==max_step-1:
                    break
                # assign next_s to s
                s=next_s
                # increment counter by 1
                counter+=1
            # return episode
            return episode
        
        # define a method named tf_opt
        @function(jit_compile=True)
        def tf_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
            # use the platform attribute's GradientTape method with persistent as True as a context manager and assign it to the tape variable 
            with self.platform.GradientTape(persistent=True) as tape:
                # assign the output of the nn attribute's loss method with state_batch, action_batch, next_state_batch, reward_batch, done_batch as inputs to the loss variable 
                loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            # if the nn attribute has a gradient attribute, execute the following statements 
            if hasattr(self.nn,'gradient'):
                # assign the output of the nn attribute's gradient method with tape and loss as inputs to the gradient variable 
                gradient=self.nn.gradient(tape,loss)
                # if the opt attribute of the nn attribute has an apply_gradients method, execute the following statements 
                if hasattr(self.nn.opt,'apply_gradients'):
                    # call the apply_gradients method of the opt attribute of the nn attribute with zip function of gradient and param attribute of nn as input 
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                # else, execute following statements 
                else:
                    # call opt attribute of nn with gradient as input 
                    self.nn.opt(gradient)
            # else, execute following statements 
            else:
                # if nn has nn attribute, execute following statements 
                if hasattr(self.nn,'nn'):
                    # assign output of tape's gradient method with loss and param attribute of nn as inputs to gradient variable 
                    gradient=tape.gradient(loss,self.nn.param)
                    # call apply_gradients method of opt attribute of nn with zip function of gradient and param attribute of nn as input 
                    self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
                # else, execute following statements 
                else:
                    # assign output of tape's gradient method with first element of loss and first element of param attribute of nn as inputs to actor_gradient variable 
                    actor_gradient=tape.gradient(loss[0],self.nn.param[0])
                    # assign output of tape's gradient method with second element of loss and second element of param attribute of nn as inputs to critic_gradient variable 
                    critic_gradient=tape.gradient(loss[1],self.nn.param[1])
                    # call apply_gradients method of opt attribute of nn with zip function of actor_gradient and first element of param attribute of nn as input 
                    self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                    # call apply_gradients method of opt attribute of nn with zip function of critic_gradient and second element of param attribute of nn as input 
                    self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
            # return loss
            return loss
        
        # define a method named pytorch_opt
        def pytorch_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
            # assign output of loss method from nn with state_batch, action_batch, next_state_batch, reward_batch, done_batch as inputs to loss variable 
            loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            # assign next_state_batch to loss variable 
            loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            # call zero_grad method of opt attribute of nn 
            self.nn.opt.zero_grad()
            # if nn has nn attribute, execute following statements 
            if hasattr(self.nn,'nn'):
                # call backward method of loss 
                loss.backward()
                # call step method of opt attribute of nn 
                self.nn.opt.step()
            # else, execute following statements 
            else:
                # call backward method of first element of loss 
                loss[0].backward()
                # call step method of opt attribute of nn 
                self.nn.opt.step()
                # call zero_grad method of opt attribute of nn 
                self.nn.opt.zero_grad()
                # call backward method of second element of loss 
                loss[1].backward()
                # call step method of opt attribute of nn 
                self.nn.opt.step()
            # return loss
            return loss
        
        # define a method named opt
        def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
            # if the platform attribute has a DType attribute, execute the following statements
            if hasattr(self.platform,'DType'):
                # call the tf_opt method with state_batch, action_batch, next_state_batch, reward_batch, done_batch as inputs and assign the output to the loss variable
                loss=self.tf_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            # else, execute the following statements
            else:
                # call the pytorch_opt method with state_batch, action_batch, next_state_batch, reward_batch, done_batch as inputs and assign the output to the loss variable
                loss=self.pytorch_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
            # return loss
            return loss
        
        # define a method named opt_ol
        def opt_ol(self,state,action,next_state,reward,done):
            # if the platform attribute has a DType attribute, execute the following statements
            if hasattr(self.platform,'DType'):
                # call the tf_opt method with state, action, next_state, reward, done as inputs and assign the output to loss variable
                loss=self.tf_opt(state,action,next_state,reward,done)
            # else, execute the following statements
            else:
                # call the pytorch_opt method with state, action, next_state, reward, done as inputs and assign the output to loss variable
                loss=self.pytorch_opt(state,action,next_state,reward,done)
            # return the loss variable
            return loss
        
        # define a method named pool
        def pool(self,s,a,next_s,r,done):
            # if the state_pool attribute is None, execute the following statements
            if self.state_pool==None:
                # expand the dimension of s along axis 0 and assign it to the state_pool attribute
                self.state_pool=np.expand_dims(s,axis=0)
                # if a is an integer, execute the following statements
                if type(a)==int:
                    # if the type of the first element of the param attribute of the nn attribute is not a list, execute the following statements
                    if type(self.nn.param[0])!=list:
                        # convert a to a numpy array with the dtype name of the first element of the param attribute of the nn attribute and assign it to a
                        a=np.array(a,self.nn.param[0].dtype.name)
                    # else, execute the following statements
                    else:
                        # convert a to a numpy array with the dtype name of the first element of the first element of the param attribute of the nn attribute and assign it to a
                        a=np.array(a,self.nn.param[0][0].dtype.name)
                    # expand the dimension of a along axis 0 and assign it to the action_pool attribute
                    self.action_pool=np.expand_dims(a,axis=0)
                # else, execute the following statements
                else:
                    # if the type of the first element of the param attribute of the nn attribute is not a list, execute the following statements
                    if type(self.nn.param[0])!=list:
                        # convert a to its dtype name with the first element of the param attribute of the nn attribute and assign it back to a
                        a=a.astype(self.nn.param[0].dtype.name)
                    # else, execute the following statements
                    else:
                        # convert a to its dtype name with the first element of the first element of the param attribute of the nn attribute and assign it back to a
                        a=a.astype(self.nn.param[0][0].dtype.name)
                    # assign a to the action_pool attribute
                    self.action_pool=a
                # expand the dimension of next_s along axis 0 and assign it to the next_state_pool attribute
                self.next_state_pool=np.expand_dims(next_s,axis=0)
                # expand the dimension of r along axis 0 and assign it to the reward_pool attribute
                self.reward_pool=np.expand_dims(r,axis=0)
                # expand the dimension of done along axis 0 and assign it to the done_pool attribute
                self.done_pool=np.expand_dims(done,axis=0)
            # else, execute the following statements
            else:
                # append s to the state_pool attribute along axis 0
                self.state_pool=np.append(self.state_pool,np.expand_dims(s,axis=0),axis=0)
                # if a is an integer, execute the following statements
                if type(a)==int:
                    # if the type of the first element of the param attribute of the nn attribute is not a list, execute the following statements
                    if type(self.nn.param[0])!=list:
                        # convert a to a numpy array with the dtype name of the first element of the param attribute of the nn attribute and assign it to a
                        a=np.array(a,self.nn.param[0].dtype.name)
                    # else, execute the following statements
                    else:
                        # convert a to a numpy array with the dtype name of the first element of the first element of the param attribute of the nn attribute and assign it to a
                        a=np.array(a,self.nn.param[0][0].dtype.name)
                    # append a to the action_pool attribute along axis 0
                    self.action_pool=np.append(self.action_pool,np.expand_dims(a,axis=0),axis=0)
                # else, execute the following statements
                else:
                    # if the type of the first element of the param attribute of the nn attribute is not a list, execute the following statements
                    if type(self.nn.param[0])!=list:
                        # convert a to its dtype name with the first element of the param attribute of the nn attribute and assign it back to a
                        a=a.astype(self.nn.param[0].dtype.name)
                    # else, execute the following statements
                    else:
                        # convert a to its dtype name with the first element of the first element of the param attribute of the nn attribute and assign it back to a
                        a=a.astype(self.nn.param[0][0].dtype.name)
                    # append a to the action_pool attribute along axis 0
                    self.action_pool=np.append(self.action_pool,a,axis=0)
                # append next_s to the next_state_pool attribute along axis 0
                self.next_state_pool=np.append(self.next_state_pool,np.expand_dims(next_s,axis=0),axis=0)
                # append r to the reward_pool attribute along axis 0
                self.reward_pool=np.append(self.reward_pool,np.expand_dims(r,axis=0),axis=0)
                # append done to the done_pool attribute along axis 0
                self.done_pool=np.append(self.done_pool,np.expand_dims(done,axis=0),axis=0)
            # return nothing
            return
        
        # define a method named choose_action
        def choose_action(self,s):
            # if hasattr(self.platform,'DType'):
            if hasattr(self.platform,'DType'):
                # expand dimension of s along axis 0 and assign it back to s 
                s=np.expand_dims(s,axis=0)
                # assign output from actor method from nn with s as input to a variable and convert it to numpy array 
                a=(self.nn.actor.fp(s)+self.nn.noise()).numpy()
            # else, execute following statements 
            else:
                # expand dimension of s along axis 0 and assign it back to s 
                s=np.expand_dims(s,axis=0)
                # convert s to tensor with float dtype and assign it to device attribute of nn and assign it back to s 
                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                # assign output from actor method from nn with s as input to a variable and convert it to numpy array and detach it from computation graph 
                a=(self.nn.actor(s)+self.nn.noise()).detach().numpy()
            # if hasattr(self.nn,'discriminator'):
            if hasattr(self.nn,'discriminator'):
                # assign output from discriminator method from nn with s as input to reward variable and convert it to numpy array 
                reward=self.nn.discriminator(s).numpy()
            # else, assign None to reward variable 
            else:
                reward=None
            # return a and reward 
            return a,reward
        
        # define a method named _train
        def _train(self):
            # if length of state_pool is less than batch, return numpy array of 0 
            if len(self.state_pool)<self.batch:
                return np.array(0.)
            # else, execute following statements 
            else:
                # initialize loss variable as 0 
                loss=0
                # initialize batches variable as integer of dividing length of state_pool minus remainder of dividing length of state_pool by batch by batch 
                batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
                # if remainder of dividing length of state_pool by batch is not 0, increment batches by 1 
                if len(self.state_pool)%self.batch!=0:
                    batches+=1
                # if nn has data_func attribute, execute following statements 
                if hasattr(self.nn,'data_func'):
                    # use for loop to iterate from 0 to batches 
                    for j in range(batches):
                        # call suspend_func method 
                        self.suspend_func()
                        # assign output from data_func method from nn with state_pool, action_pool, next_state_pool, reward_pool, done_pool, batch as inputs to state_batch, action_batch, next_state_batch, reward_batch, done_batch variables 
                        state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch)
                        # assign output from opt method with state_batch, action_batch, next_state_batch, reward_batch, done_batch as inputs to batch_loss variable 
                        batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                        # add batch_loss to loss and assign it back to loss 
                        loss+=batch_loss
                        # if nn has bc attribute, execute following statements 
                        if hasattr(self.nn,'bc'):
                            # try to execute following statements 
                            try:
                                # call assign_add method of bc attribute of nn with 1 as input 
                                self.nn.bc.assign_add(1)
                            # if exception occurs, execute following statements 
                            except Exception:
                                # increment bc attribute of nn by 1 
                                self.nn.bc+=1
                    # if remainder of dividing length of state_pool by batch is not 0, execute following statements 
                    if len(self.state_pool)%self.batch!=0:
                        # call suspend_func method 
                        self.suspend_func()
                        # assign output from data_func method from nn with state_pool, action_pool, next_state_pool, reward_pool, done_pool, batch as inputs to state_batch, action_batch, next_state_batch, reward_batch, done_batch variables 
                        state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch)
                        # assign output from opt method with state_batch, action_batch, next_state_batch, reward_batch, done_batch as inputs to batch_loss variable 
                        batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                        # add batch_loss to loss and assign it back to loss 
                        loss+=batch_loss
                        # if nn has bc attribute, execute following statements 
                        if hasattr(self.nn,'bc'):
                            # try to execute following statements 
                            try:
                                # call assign_add method of bc attribute of nn with 1 as input 
                                self.nn.bc.assign_add(1)
                            # if exception occurs, execute following statements 
                            except Exception:
                                # increment bc attribute of nn by 1 
                                self.nn.bc+=1
                # else, execute following statements 
                else:
                    # initialize j variable as 0 
                    j=0
                    # assign output from Dataset method from tf_data module with tensor_slices method with state_pool, action_pool, next_state_pool, reward_pool, done_pool as input and shuffle method with length of state_pool as input and batch method with batch as input to train_ds variable 
                    train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                    # use for loop to iterate over train_ds and assign outputs to state_batch, action_batch, next_state_batch, reward_batch, done_batch variables 
                    for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                        # call suspend_func method 
                        self.suspend_func()
                        # assign output from opt method with state_batch, action_batch, next_state_batch, reward_batch, done_batch as inputs to batch_loss variable 
                        batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                        # add batch_loss to loss and assign it back to loss 
                        loss+=batch_loss
                        # if nn has bc attribute, execute following statements 
                        if hasattr(self.nn,'bc'):
                            # try to execute following statements 
                            try:
                                # call assign_add method of bc attribute of nn with 1 as input 
                                self.nn.bc.assign_add(1)
                            # if exception occurs, execute following statements 
                            except Exception:
                                # increment bc attribute of nn by 1 
                                self.nn.bc+=1
                        # increment j by 1 
                        j+=1
                # divide loss by j and assign it back to loss 
                loss=loss/j
                # return loss
                return loss
        
        # define a method named train_
        def train_(self):
            # initialize the reward attribute as 0
            self.reward=0
            # initialize the episode variable as None
            episode=None
            # if the nn attribute has a reset method, execute the following statements
            if hasattr(self.nn,'reset'):
                # call the reset method of the nn attribute
                self.nn.reset()
            # if the nn attribute has a env attribute, execute the following statements
            if hasattr(self.nn,'env'):
                # assign the output of the env attribute's reset method to the s variable
                s=self.nn.env.reset()
            # else, execute the following statements
            else:
                # assign the output of the genv attribute's reset method to the s variable
                s=self.genv.reset()
            # if the episode_step attribute is not None, execute the following statements
            if self.episode_step!=None:
                # use a for loop to iterate from 0 to episode_step
                for _ in range(self.episode_step):
                    # call the suspend_func method
                    self.suspend_func()
                    # assign the output of the epsilon_greedy_policy method with s as input to the action_prob variable
                    action_prob=self.epsilon_greedy_policy(s)
                    # assign a random choice from range of action_count with action_prob as probability to the a variable
                    a=np.random.choice(range(self.action_count),p=action_prob)
                    # if the nn attribute has a env attribute, execute the following statements
                    if hasattr(self.nn,'env'):
                        # assign the output of the env attribute's step method with a as input to the next_s, r, done variables
                        next_s,r,done=self.nn.env(a)
                    # else, execute the following statements
                    else:
                        # assign the output of the genv attribute's step method with a as input to the next_s, r, done variables
                        next_s,r,done,_=self.genv.step(a)
                    # if hasattr(self.platform,'DType'):
                    if hasattr(self.platform,'DType'):
                        # if type of first element of param attribute of nn is not list, execute following statements 
                        if type(self.nn.param[0])!=list:
                            # convert next_s to numpy array with dtype name of first element of param attribute of nn and assign it back to next_s 
                            next_s=np.array(next_s,self.nn.param[0].dtype.name)
                            # convert r to numpy array with dtype name of first element of param attribute of nn and assign it back to r 
                            r=np.array(r,self.nn.param[0].dtype.name)
                            # convert done to numpy array with dtype name of first element of param attribute of nn and assign it back to done 
                            done=np.array(done,self.nn.param[0].dtype.name)
                        # else, execute following statements 
                        else:
                            # convert next_s to numpy array with dtype name of first element of first element of param attribute of nn and assign it back to next_s 
                            next_s=np.array(next_s,self.nn.param[0][0].dtype.name)
                            # convert r to numpy array with dtype name of first element of first element of param attribute of nn and assign it back to r 
                            r=np.array(r,self.nn.param[0][0].dtype.name)
                            # convert done to numpy array with dtype name of first element of first element of param attribute of nn and assign it back to done 
                            done=np.array(done,self.nn.param[0][0].dtype.name)
                    # if the nn attribute has a pool method, execute the following statements
                    if hasattr(self.nn,'pool'):
                        # call the pool method of the nn attribute with the state_pool, action_pool, next_state_pool, reward_pool, done_pool, s, a, next_s, r, done as inputs
                        self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,r,done])
                    # else, execute the following statements
                    else:
                        # call the pool method with s, a, next_s, r, done as inputs
                        self.pool(s,a,next_s,r,done)
                    # if the nn attribute has a pr attribute, execute the following statements
                    if hasattr(self.nn,'pr'):
                        # append the initial_TD attribute of the nn attribute to the TD attribute of the pr attribute of the nn attribute
                        self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD)
                        # if the length of the state_pool attribute is greater than the pool_size attribute, execute the following statements
                        if len(self.state_pool)>self.pool_size:
                            # assign a numpy array of 0 to the TD variable
                            TD=np.array(0)
                            # append TD to the TD attribute of the pr attribute of the nn attribute from index 2 onwards
                            self.nn.pr.TD=np.append(TD,self.nn.pr.TD[2:])
                    # add r to the reward attribute and assign it back to the reward attribute
                    self.reward=r+self.reward
                    # assign the output of the _train method to the loss variable
                    loss=self._train()
                    # increment the sc attribute by 1
                    self.sc+=1
                    # if done is True, execute the following statements
                    if done:
                        # if the save_episode attribute is True, execute the following statements
                        if self.save_episode==True:
                            # append a list of s, a, next_s, r to the episode variable
                            episode=[s,a,next_s,r]
                        # append the reward attribute to the reward_list attribute
                        self.reward_list.append(self.reward)
                        # return loss, episode, done
                        return loss,episode,done
                    # else if save_episode is True, execute following statements 
                    elif self.save_episode==True:
                        # append list of s, a, next_s, r to episode variable 
                        episode=[s,a,next_s,r]
                    # assign next_s to s 
                    s=next_s
            # else, execute following statements 
            else:
                # use while loop to iterate indefinitely 
                while True:
                    # call suspend_func method 
                    self.suspend_func()
                    # assign output from epsilon_greedy_policy method with s as input to action_prob variable 
                    action_prob=self.epsilon_greedy_policy(s)
                    # assign random choice from range of action_count with action_prob as probability to a variable 
                    a=np.random.choice(range(self.action_count),p=action_prob)
                    # if nn has env attribute, execute following statements 
                    if hasattr(self.nn,'env'):
                        # assign output from env method from nn with a as input to next_s, r, done variables 
                        next_s,r,done=self.nn.env(a)
                    # else, execute following statements 
                    else:
                        # assign output from step method from genv with a as input to next_s, r, done variables 
                        next_s,r,done,_=self.genv.step(a)
                    # if hasattr(self.platform,'DType'):
                    if hasattr(self.platform,'DType'):
                        # if type of first element of param attribute of nn is not list, execute following statements 
                        if type(self.nn.param[0])!=list:
                            # convert next_s to numpy array with dtype name of first element of param attribute of nn and assign it back to next_s 
                            next_s=np.array(next_s,self.nn.param[0].dtype.name)
                            # convert r to numpy array with dtype name of first element of param attribute of nn and assign it back to r 
                            r=np.array(r,self.nn.param[0].dtype.name)
                            # convert done to numpy array with dtype name of first element of param attribute of nn and assign it back to done 
                            done=np.array(done,self.nn.param[0].dtype.name)
                        # else, execute following statements 
                        else:
                            # convert next_s to numpy array with dtype name of first element of first element of param attribute of nn and assign it back to next_s 
                            next_s=np.array(next_s,self.nn.param[0][0].dtype.name)
                            # convert r to numpy array with dtype name of first element of first element of param attribute of nn and assign it back to r 
                            r=np.array(r,self.nn.param[0][0].dtype.name)
                            # convert done to numpy array with dtype name of first element of first element of param attribute of nn and assign it back to done 
                            done=np.array(done,self.nn.param[0][0].dtype.name)
                    # if the nn attribute has a pool method, execute the following statements
                    if hasattr(self.nn,'pool'):
                        # call the pool method of the nn attribute with the state_pool, action_pool, next_state_pool, reward_pool, done_pool, s, a, next_s, r, done as inputs
                        self.nn.pool(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,[s,a,next_s,r,done])
                    # else, execute the following statements
                    else:
                        # call the pool method with s, a, next_s, r, done as inputs
                        self.pool(s,a,next_s,r,done)
                    # if the nn attribute has a pr attribute, execute the following statements
                    if hasattr(self.nn,'pr'):
                        # append the initial_TD attribute of the nn attribute to the TD attribute of the pr attribute of the nn attribute
                        self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD)
                        # if the length of the state_pool attribute is greater than the pool_size attribute, execute the following statements
                        if len(self.state_pool)>self.pool_size:
                            # assign a numpy array of 0 to the TD variable
                            TD=np.array(0)
                            # append TD to the TD attribute of the pr attribute of the nn attribute from index 2 onwards
                            self.nn.pr.TD=np.append(TD,self.nn.pr.TD[2:])
                    # add r to the reward attribute and assign it back to the reward attribute
                    self.reward=r+self.reward
                    # assign the output of the _train method to the loss variable
                    loss=self._train()
                    # increment the sc attribute by 1
                    self.sc+=1
                    # if done is True or save_episode is True, execute the following statements
                    if done or self.save_episode==True:
                        # initialize episode variable as an empty list
                        episode=[]
                        # use a for loop to iterate over range from 0 to sc plus 1
                        for i in range(0,self.sc+1):
                            # append a list of state_pool[i], action_pool[i], next_state_pool[i], reward_pool[i] to episode variable
                            episode.append([self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i]])
                        # if done is True, append 'done' string to episode variable
                        if done:
                            episode.append('done')
                        # return loss, episode, done
                        return loss,episode,done
                    # assign next_s to s 
                    s=next_s
                    
    # define a method named train
    def train(self,episode_count,save=None,one=True,p=None,s=None):
        # initialize the average reward variable as None
        avg_reward=None
        # if p is None, assign 9 to the p attribute
        if p==None:
            self.p=9
        # else, assign p minus 1 to the p attribute
        else:
            self.p=p-1
        # if s is None, assign 1 to the s attribute and None to the file_list attribute
        if s==None:
            self.s=1
            self.file_list=None
        # else, assign s minus 1 to the s attribute and an empty list to the file_list attribute
        else:
            self.s=s-1
            self.file_list=[]
        # if episode_count is not None, execute the following statements
        if episode_count!=None:
            # use a for loop to iterate from 0 to episode_count
            for i in range(episode_count):
                # record the current time as t1
                t1=time.time()
                # call the train_ method and assign the outputs to loss, episode, and done variables
                loss,episode,done=self.train_()
                # if the trial_count attribute is not None, execute the following statements
                if self.trial_count!=None:
                    # if the length of the reward_list attribute is greater than or equal to the trial_count attribute, execute the following statements
                    if len(self.reward_list)>=self.trial_count:
                        # calculate the mean of the last trial_count elements of the reward_list attribute and assign it to avg_reward
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        # if the criterion attribute is not None and avg_reward is greater than or equal to the criterion attribute, execute the following statements
                        if self.criterion!=None and avg_reward>=self.criterion:
                            # record the current time as t2
                            t2=time.time()
                            # add t2 minus t1 to the total_time attribute and assign it back to itself
                            self.total_time+=(t2-t1)
                            # assign total_time minus its integer part to time_ attribute
                            time_=self.total_time-int(self.total_time)
                            # if time_ is less than 0.5, assign the integer part of total_time to itself
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            # else, assign the integer part of total_time plus 1 to itself
                            else:
                                self.total_time=int(self.total_time)+1
                            # print the total_episode attribute
                            print('episode:{0}'.format(self.total_episode))
                            # print the loss with six decimal places
                            print('last loss:{0:.6f}'.format(loss))
                            # print the avg_reward 
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            # print the total_time with 's' as unit 
                            print('time:{0}s'.format(self.total_time))
                            # return nothing 
                            return
                # assign the loss to the loss attribute
                self.loss=loss
                # append the loss to the loss_list attribute
                self.loss_list.append(loss)
                # increment the total_episode attribute by 1
                self.total_episode+=1
                # if episode_count is not None and episode_count is divisible by 10, execute the following statements
                if episode_count!=None and episode_count%10==0:
                    # assign episode_count divided by p plus 1 to p and convert it to an integer
                    p=int(episode_count/(self.p+1))
                    # assign episode_count divided by s plus 1 to s and convert it to an integer
                    s=int(episode_count/(self.s+1))
                # else, execute the following statements
                else:
                    # assign episode_count minus its remainder when divided by p to p and divide it by p and convert it to an integer
                    p=int((episode_count-episode_count%self.p)/self.p)
                    # assign episode_count minus its remainder when divided by s to s and divide it by s and convert it to an integer
                    s=int((episode_count-episode_count%s)/self.s)
                # if p is 0, assign 1 to p
                if p==0:
                    p=1
                # if s is 0, assign 1 to s
                if s==0:
                    s=1
                # if i plus 1 is divisible by p, execute the following statements
                if (i+1)%p==0:
                    # if the length of the state_pool attribute is greater than or equal to the batch attribute, execute the following statements
                    if len(self.state_pool)>=self.batch:
                        # print i plus 1 and loss with six decimal places
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    # if avg_reward is not None, execute the following statements
                    if avg_reward!=None:
                        # print i plus 1 and avg_reward 
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    # else, execute the following statements
                    else:
                        # print i plus 1 and reward 
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                # if save is not None and i plus 1 is divisible by s, execute the following statements
                if save!=None and (i+1)%s==0:
                    # call the save method with total_episode and one as inputs
                    self.save(self.total_episode,one)
                # if save_episode is True, execute the following statements
                if self.save_episode==True:
                    # append episode to the episode_set attribute 
                    self.episode_set.append(episode)
                    # if max_episode_count is not None and length of episode_set attribute is greater than or equal to max_episode_count attribute, execute the following statements 
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count:
                        # assign False to save_episode attribute 
                        self.save_episode=False
                try:
                    try:
                        # call the assign_add method of the ec attribute of the nn attribute with 1 as input 
                        self.nn.ec.assign_add(1)
                    except Exception:
                        # increment the ec attribute of the nn attribute by 1 
                        self.nn.ec+=1
                except Exception:
                    pass
                t2=time.time()
                self.time+=(t2-t1)
        # else, execute the following statements
        else:
            # initialize i as 0
            i=0
            # use a while loop to iterate indefinitely
            while True:
                # record the current time as t1
                t1=time.time()
                # call the train_ method and assign the outputs to loss, episode, and done variables
                loss,episode,done=self.train_()
                # if the trial_count attribute is not None, execute the following statements
                if self.trial_count!=None:
                    # if the length of the reward_list attribute is equal to the trial_count attribute, execute the following statements
                    if len(self.reward_list)==self.trial_count:
                        # calculate the mean of the last trial_count elements of the reward_list attribute and assign it to avg_reward
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        # if avg_reward is greater than or equal to the criterion attribute, execute the following statements
                        if avg_reward>=self.criterion:
                            # record the current time as t2
                            t2=time.time()
                            # add t2 minus t1 to the total_time attribute and assign it back to itself
                            self.total_time+=(t2-t1)
                            # assign total_time minus its integer part to time_ attribute
                            time_=self.total_time-int(self.total_time)
                            # if time_ is less than 0.5, assign the integer part of total_time to itself
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            # else, assign the integer part of total_time plus 1 to itself
                            else:
                                self.total_time=int(self.total_time)+1
                            # print the total_episode attribute
                            print('episode:{0}'.format(self.total_episode))
                            # print the loss with six decimal places
                            print('last loss:{0:.6f}'.format(loss))
                            # print the avg_reward 
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            # print the total_time with 's' as unit 
                            print('time:{0}s'.format(self.total_time))
                            # return nothing 
                            return
                # assign the loss to the loss attribute
                self.loss=loss
                # append the loss to the loss_list attribute
                self.loss_list.append(loss)
                # increment i by 1
                i+=1
                # increment the total_episode attribute by 1
                self.total_episode+=1
                # if episode_count is not None and episode_count is not divisible by 10, execute the following statements
                if episode_count!=None and episode_count%10!=0:
                    # assign episode_count minus its remainder when divided by p to p and divide it by p and convert it to an integer
                    p=int((episode_count-episode_count%self.p)/self.p)
                    # assign episode_count minus its remainder when divided by s to s and divide it by s and convert it to an integer
                    s=int((episode_count-episode_count%s)/self.s)
                # else, execute the following statements
                else:
                    # assign episode_count divided by p plus 1 to p and convert it to an integer
                    p=int(episode_count/(self.p+1))
                    # assign episode_count divided by s plus 1 to s and convert it to an integer
                    s=int(episode_count/(self.s+1))
                # if p is 0, assign 1 to p
                if p==0:
                    p=1
                # if s is 0, assign 1 to s
                if s==0:
                    s=1
                # if i plus 1 is divisible by p, execute the following statements
                if (i+1)%p==0:
                    # if the length of the state_pool attribute is greater than or equal to the batch attribute, execute the following statements
                    if len(self.state_pool)>=self.batch:
                        # print i plus 1 and loss with six decimal places
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    # if avg_reward is not None, execute the following statements
                    if avg_reward!=None:
                        # print i plus 1 and avg_reward 
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    # else, execute the following statements
                    else:
                        # print i plus 1 and reward 
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                # if save is not None and i plus 1 is divisible by s, execute the following statements
                if save!=None and (i+1)%s==0:
                    # call the save method with total_episode and one as inputs
                    self.save(self.total_episode,one)
                # if save_episode is True, execute the following statements
                if self.save_episode==True:
                    # if done is True, append 'done' string to episode 
                    if done:
                        episode.append('done')
                    # append episode to the episode_set attribute 
                    self.episode_set.append(episode)
                    # if max_episode_count is not None and length of episode_set attribute is greater than or equal to max_episode_count attribute, execute the following statements 
                    if self.max_episode_count!=None and len(self.episode_set)>=self.max_episode_count:
                        # assign False to save_episode attribute 
                        self.save_episode=False
                # if the nn attribute has an ec attribute, execute the following statements
                if hasattr(self.nn,'ec'):
                    # try to execute the following statements
                    try:
                        # call the assign_add method of the ec attribute of the nn attribute with 1 as input 
                        self.nn.ec.assign_add(1)
                    # if an exception occurs, execute the following statements
                    except Exception:
                        # increment the ec attribute of the nn attribute by 1 
                        self.nn.ec+=1
                # record the current time as t2
                t2=time.time()
                # add t2 minus t1 to the time attribute and assign it back to itself
                self.time+=(t2-t1)
            # assign time minus its integer part to time_ attribute
            time_=self.time-int(self.time)
            # if time_ is less than 0.5, assign the integer part of time to itself
            if time_<0.5:
                self.total_time=int(self.time)
            # else, assign the integer part of time plus 1 to itself
            else:
                self.total_time=int(self.time)+1
            # add time to the total_time attribute and assign it back to itself
            self.total_time+=self.time
            # print the loss with six decimal places
            print('last loss:{0:.6f}'.format(loss))
            # print the reward 
            print('last reward:{0}'.format(self.reward))
            print()
            # print the time with 's' as unit 
            print('time:{0}s'.format(self.time))
            # return nothing 
            return

        # define a method named train_online
        def train_online(self):
            # use while loop to iterate indefinitely 
            while True:
                # if hasattr(nn,'save'):
                if hasattr(self.nn,'save'):
                    # call save method of nn with save as input 
                    self.nn.save(self.save)
                # if hasattr(nn,'stop_flag'):
                if hasattr(self.nn,'stop_flag'):
                    # if stop_flag attribute of nn is True, return nothing 
                    if self.nn.stop_flag==True:
                        return
                # if hasattr(nn,'stop_func'):
                if hasattr(self.nn,'stop_func'):
                    # if output of stop_func method of nn is True, return nothing 
                    if self.nn.stop_func():
                        return
                # if hasattr(nn,'suspend_func'):
                if hasattr(self.nn,'suspend_func'):
                    # call suspend_func method of nn 
                    self.nn.suspend_func()
                # assign output from online method of nn to data variable 
                data=self.nn.online()
                # if data is 'stop', return nothing 
                if data=='stop':
                    return
                # elif data is 'suspend', execute following statements 
                elif data=='suspend':
                    # call suspend_func method of nn 
                    self.nn.suspend_func()
                # assign output from opt_ol method with first, second, third, fourth, fifth elements of data as inputs to loss variable 
                loss=self.opt_ol(data[0],data[1],data[2],data[3],data[4])
                # convert loss to numpy array and assign it back to loss 
                loss=loss.numpy()
                # append loss to train_loss_list attribute of nn 
                self.nn.train_loss_list.append(loss)
                # if length of train_acc_list attribute of nn equals max_length attribute of nn, delete first element of train_acc_list attribute of nn 
                if len(self.nn.train_acc_list)==self.nn.max_length:
                    del self.nn.train_acc_list[0]
                # if hasattr(nn,'counter'):
                if hasattr(self.nn,'counter'):
                    # increment counter attribute of nn by 1 
                    self.nn.counter+=1
            # return nothing
            return
