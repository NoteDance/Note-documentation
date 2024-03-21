# Non-parallel training:
## DL:
### Save and restore:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train) #input train data to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
kernel.save()                #save the neural network to a file
```
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
kernel=k.kernel()             #create kernel object without a network
kernel.platform=tf            #set the platform to tensorflow
kernel.data(x_train,y_train)  #input train data to the kernel
kernel.restore('save.dat')    #restore the network from a file
kernel.train(32,1)            #train the network again with batch size 32 and epoch 1
```

## Training with test data
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn_acc as n   #import neural network module with accuracy function
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train,x_test,y_test) #input train and test data and labels to the kernel
kernel.train(32,5,32)        #train the network with batch size 32, epoch 5 and test batch size 32
kernel.test(x_test,y_test,32)#test the network performance on the test set with batch size 32
```

### Saving multiple files in training:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train) #input train data to the kernel
kernel.train(32,5,save=True,one=False,s=3)           #train the network with batch size 32 and epoch 5
```

### Stop training and saving when condition is met:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.stop=True             #set the flag to stop training when a condition is met
kernel.end_loss=0.7          #set the condition to stop training when the loss is less than 0.7
kernel.data(x_train,y_train) #input train data to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
```

### Visualization:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.stop=True             #set the flag to stop training when a condition is met
kernel.end_loss=0.7          #set the condition to stop training when the loss is less than 0.7
kernel.data(x_train,y_train) #input train data to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
kernel.visualize_train()     #visualize the loss
```

### Set the print count:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train) #input train data to the kernel
kernel.train(32,5,p=3)           #train the network with batch size 32 and epoch 5
```


## RL:
### Saving multiple files in training:
```python
import Note.RL.kernel as k   #import kernel module
import tensorflow as tf           #import tensorflow library
import neuralnetwork.RL.tensorflow.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)                #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)              #create kernel object with the network
kernel.platform=tf                #set the platform to tensorflow
kernel.action_count=2             #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.train(100,save=True,one=False,s=3)                 #train the network for 100 episodes
```

### Stop training and saving when condition is met:
```python
import Note.RL.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)       #create kernel object with the network
kernel.stop=True             #set the flag to stop training when a condition is met
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200) #set up the hyperparameters for training and the condition to stop training when the average reward of 10 trials is greater than 200
kernel.train(100)            #train the network for 500 episodes
```

### Visualization:
```python
import Note.RL.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)       #create kernel object with the network
kernel.stop=True             #set the flag to stop training when a condition is met
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200) #set up the hyperparameters for training and the condition to stop training when the average reward of 10 trials is greater than 200
kernel.train(100)            #train the network for 500 episodes
kernel.visualize_reward()    #visualize the reward
kernel.visualize_train()     #visualize the loss
```

### Set the print count:
```python
import Note.RL.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)       #create kernel object with the network
kernel.stop=True             #set the flag to stop training when a condition is met
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200) #set up the hyperparameters for training and the condition to stop training when the average reward of 10 trials is greater than 200
kernel.train(100,p=3)            #train the network for 500 episodes
```


## Parallel test:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
nn.build()                   #build the network structure
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.process_t=3           #set the number of processes to test
kernel.data(x_train,y_train,x_test,y_test) #input train and test data to the kernel
kernel.train(32,5,32)        #train the network with batch size 32, epoch 5 and test batch size 32
```


# Parallel training:
## DL:
### PO1:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=1                          #use PO1 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### PO2:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
x_train=x_train.reshape([60000,784]) #reshape data to fit the network input
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=2                          #use PO2 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
g_lock=Lock()                        #create a global lock for gradient computing
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock,g_lock)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
x_train=x_train.reshape([60000,784]) #reshape data to fit the network input
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=2                          #use PO2 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
g_lock=[Lock(),Lock()]               #create a list of global locks for gradient computing
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock,g_lock)).start() #start each process with the train function and pass the process id, the locks and the global locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### PO3：
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=Lock()                          #create a lock for synchronization
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock)).start() #start each process with the train function and pass the process id and the lock as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### Save and restore:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
kernel.save()                        #save the neural network to a file
```
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
kernel=k.kernel()                    #create kernel object without a network
kernel.process=3                     #set the number of processes to train
kernel.epoch=1                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
kernel.restore('save.dat',manager)           #restore the neural network from a file
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
```

### Visualization:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
kernel.visualize_train()             #visualize the loss
```

### Parallel test:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.process_t=3                   #set the number of processes to test
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train,x_test,y_test) #input train and test data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,None,None,32)).start() #start each process with the train function and pass the process id, the locks, the test flag and the test batch size as arguments
```

### Saving multiple files in parallel training:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.s=3                           #set the maximum number of saved files
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id and the lock as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### Stop training and saving when condition is met:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
x_train=x_train.reshape([60000,784]) #reshape data to fit the network input
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.stop=True                     #set the flag to stop training when a condition is met
kernel.end_loss=0.7                  #set the condition to stop training when the loss is less than 0.7
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as arguments
```

### Process priority:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.priority_flag=True            #set the flag to use priority scheduling for processes
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### Gradient attenuation:
**Calculate the attenuation coefficient based on the optimization counter using the attenuation function.**

```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn_attenuate as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.priority_flag=True            #set the flag to use priority scheduling for processes
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```


## RL：
**Pool Network:**
### PO1:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=1                  #use PO1 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create a list of locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```

### PO2:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=2                  #use PO2 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create a list of locks for synchronization
g_lock=Lock()                #create a global lock for gradient computing
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock,g_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks, the pool locks and the global lock as arguments
```

### PO3:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create three locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```

### Visualization:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock()]  #create three locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
kernel.visualize_reward()    #visualize the reward
kernel.visualize_train()     #visualize the loss
```

### Saving multiple files in parallel training:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
kernel.s=3                   #set the maximum number of saved files
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock()]         #create two locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```

### Stop training and saving when condition is met:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.stop=True             #set the flag to stop training when a condition is met
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create three locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```

### Process priority:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.priority_flag=True    #set the flag to use priority scheduling for processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock()]  	     #create two locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```


# Neural network(non-parallel):
**ConvNeXtV2:**

**Train:**
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
from Note.nn.neuralnetwork.non_parallel.ConvNeXtV2 import ConvNeXtV2 #import neural network class
from tensorflow.keras import datasets
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
train_images,test_images=train_images/255.0,test_images/255.0
convnext_atto=ConvNeXtV2(model_type='atto',classes=10)  #create neural network object
convnext_atto.build()                           #build the network structure
kernel=k.kernel(convnext_atto)                  #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(train_images,train_labels)         #input train data to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
```
**Use the trained model:**
```python
convnext_atto.km=0
output=convnext_atto.fp(data)
```
**ConvNeXtV2:**

**Train:**
```python
import Note.DL.kernel as k   #import kernel module
from Note.nn.neuralnetwork.ConvNeXtV2 import ConvNeXtV2 #import neural network class
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)  #create neural network object
convnext_atto.build()                           #build the network structure
kernel=k.kernel(convnext_atto)                  #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(train_images,train_labels)         #input train data to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
```
**Fine tuning:**
```python
convnext_atto.fine_tuning(10,0.0001)
kernel.data(fine_tuning_data,fine_tuning_labels)
kernel.train(32,1)           #train the network with batch size 32 and epoch 1
```
**Use the trained model:**
```python
convnext_atto.fine_tuning(flag=1)
convnext_atto.km=0
output=convnext_atto.fp(data)
```


# Parallel test:
```python
import tensorflow as tf       #import tensorflow library
from multiprocessing import Process #import multiprocessing tools
import neuralnetwork.DL.tensorflow.parallel.nn_acc as n   #import neural network module with accuracy function
import Note.DL.dl.test as t   #import parallel test module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                     #create neural network object
nn.build()                    #build the network structure
test=t.parallel_test(nn,x_test,y_test,6,32) #create parallel test object with the network, the test data, the number of processes and the batch size
test.segment_data()           #segment data for each process
for p in range(6):            #loop over the processes
	Process(target=test.test).start() #start each process with the test function
loss,acc=test.loss_acc()          #calculate the loss and accuracy of the test
```


# Online training:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn_ol as n   #import neural network module with online learning
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn(x_train,y_train)     #create neural network object with train data and labels
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train) #input train data and labels to the kernel
kernel.train_online()        #train the network online
```


# Check neural network:
## DL:
You can test it before using the kernel training neural network.
```python
from Note.DL.dl.check_nn import check #import check function from check_nn module
import tensorflow as tf                #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist          #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                              #create neural network object
check(nn,tf,x_train[:32],y_train[:32]) #check the network with tensorflow platform and a sample of 32 data points and labels
```

## RL:
You can test it before using the kernel training neural network.
```python
from Note.RL.rl.check_nn import check #import check function from check_nn module
import tensorflow as tf                #import tensorflow library
import neuralnetwork.RL.tensorflow.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)                     #create neural network object with 4 inputs, 128 hidden units and 2 outputs
check(dqn,tf,2)                        #check the network with tensorflow platform and 2 actions
```


# Note.nn.initializer.initializer:
This function returns a TensorFlow variable.


# Note.nn.initializer.initializer_:
This function returns a TensorFlow variable and stores the variable in Module.param.


# Note.nn.Layers.Layers:
This class is used similarly to the tf.keras.Sequential class.


# Module:
You can initialize the param list by calling the init function of the Module class, which can clear the neural network parameters stored in the param list.
```python
Module.init()
```
