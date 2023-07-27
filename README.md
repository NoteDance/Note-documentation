# Non-parallel training:

## Save and restore:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import nn as n               #import neural network module
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

## Stop training and saving when condition is met:
### DL:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import nn as n               #import neural network module
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

### RL:
```python
import Note.RL.kernel as k   #import kernel module
import DQN as d              #import deep Q-network module
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 threads to train
kernel.stop=True             #set the flag to stop training when a condition is met
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200) #set up the hyperparameters for training and the condition to stop training when the average reward of 10 trials is greater than 200
kernel.train(500)            #train the network for 500 episodes
kernel.visualize_train()
kernel.visualize_reward()
```

## Training with test data
https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/nn_acc.py
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import nn_acc as n           #import neural network module with accuracy function
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

## Parallel test:
**You can get neural network example from the link below, and then you can import neural network and train with kernel, example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/parallel/nn.py

```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import nn as n               #import neural network module
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
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=1                          #use PO1 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### PO2:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
x_train=x_train.reshape([60000,784]) #reshape data to fit the network input
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=2                          #use PO2 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
g_lock=Lock()                        #create a global lock for gradient computing
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock,g_lock)).start() #start each process with the train function and pass the process id and locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
x_train=x_train.reshape([60000,784]) #reshape data to fit the network input
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=2                          #use PO2 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=[Lock(),Lock()]                 #create two locks for synchronization
g_lock=[Lock(),Lock()]               #create a list of global locks for gradient computing
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock,g_lock)).start() #start each process with the train function and pass the process id, the locks and the global locks as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### PO3：
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
lock=Lock()                          #create a lock for synchronization
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,lock)).start() #start each process with the train function and pass the process id and the lock as arguments
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

### Save and restore:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
kernel.save()                        #save the neural network to a file
```
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
kernel=k.kernel()                    #create kernel object without a network
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=1                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
kernel.restore('save.dat')           #restore the neural network from a file
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
```

### Parallel test:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=7                     #set the number of processes to train
kernel.process_t=3                   #set the number of processes to test
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train,x_test,y_test) #input train and test data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,None,None,32)).start() #start each process with the train function and pass the process id, the locks, the test flag and the test batch size as arguments
```

### Stop training and saving when condition is met:
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import nn as n                       #import neural network module
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
kernel.process=7                     #set the number of processes to train
kernel.data_segment_flag=True        #set the flag to segment data for each process
kernel.epoch=6                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(7):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id and locks as arguments
```

## RL：
**Pool Network:**

**You can get neural network example from the link below, and then you can import neural network and train with kernel, example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/RL/neural%20network/tensorflow/parallel/DQN.py

### PO1:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import DQN as d              #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=1                  #use PO1 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create a list of locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```

### PO2:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import DQN as d              #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=2                  #use PO2 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create a list of locks for synchronization
g_lock=Lock()                #create a global lock for gradient computing
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,100,lock,pool_lock,g_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks, the pool locks and the global lock as arguments
```

### PO3:
```python
import Note.RL.parallel.kernel as k   #import kernel module
import DQN as d              #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock(),Lock()]  #create three locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```


# Parallel test:
**You can get neural network example from the link below, and then you can import neural network and train with kernel, example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/parallel/nn.py

```python
import tensorflow as tf       #import tensorflow library
from multiprocessing import Process #import multiprocessing tools
import nn_acc as n            #import neural network module with accuracy function
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
loss=test.loss_acc()          #calculate the loss and accuracy of the test
```


# Online training:
**You can get neural network example from the link below, and then you can import neural network and train with kernel, example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0/Note%207.0%20documentation/DL/neural%20network/tensorflow/non-parallel/nn_ol.py

**example:**
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import nn_ol as n            #import neural network module with online learning
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
import nn as n                         #import neural network module
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
import DQN as d                        #import deep Q-network module
dqn=d.DQN(4,128,2)                     #create neural network object with 4 inputs, 128 hidden units and 2 outputs
check(dqn,tf,2)                        #check the network with tensorflow platform and 2 actions
```


# Note Compiler:
documentation:https://github.com/NoteDancing/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/compiler
```python
import Note.nc as nc
c=nc.compiler('nn.n')
c.Compile()
```
