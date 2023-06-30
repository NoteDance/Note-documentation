# Stop training and saving when condition is met:
## DL:
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.stop=True
kernel.end_loss=0.7
kernel.data(x_train,y_train)   #input train data
kernel.train(32,5)         #train neural network
                           #batch size:32
                           #epoch:5
```

## RL:
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel
kernel.stop=True
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200)
kernel.train(500)
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```


# Parallel test:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/process/nn.py

```python
import tensorflow as tf
import nn_acc as n
import Note.DL.dl.test as t
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()
nn.build()
test=t.parallel_test(nn,x_test,y_test,6,32)
test.segment_data()
for p in range(6):
	Process(target=test.test).start()
loss=test.loss_acc()
```


# Multiprocessing:
## DL:
### PO2:
```python
import Note.DL.process.kernel as k   #import kernel
import tensorflow as tf
import nn as n                          #import neural network
from multiprocessing import Process,Lock,Manager
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
x_train=x_train.reshape([60000,784])
nn=n.nn()                                #create neural network object
nn.build()
kernel=k.kernel(nn)   #start kernel
kernel.process=7      #7 processes to train
kernel.data_segment_flag=True
kernel.epoch=6                #epoch:6
kernel.batch=32            #batch:32
kernel.PO=2                    #use PO3
kernel.data(x_train,y_train)   #input train data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=[Lock(),Lock()]
g_lock=Lock()
for p in range(7):
	Process(target=kernel.train,args=(p,lock,g_lock)).start()
kernel.update_nn_param()
kernel.test(x_train,y_train,32)
```
```python
import Note.DL.process.kernel as k   #import kernel
import tensorflow as tf
import nn as n                          #import neural network
from multiprocessing import Process,Lock,Manager
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
x_train=x_train.reshape([60000,784])
nn=n.nn()                                #create neural network object
nn.build()
kernel=k.kernel(nn)   #start kernel
kernel.process=7      #7 processes to train
kernel.data_segment_flag=True
kernel.epoch=6                #epoch:6
kernel.batch=32            #batch:32
kernel.PO=2                    #use PO3
kernel.data(x_train,y_train)   #input train data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=[Lock(),Lock()]
g_lock=Lock()
for p in range(7):
	Process(target=kernel.train,args=(p,lock,g_lock)).start()
kernel.update_nn_param()
kernel.test(x_train,y_train,32)
```
### Stop training and saving when condition is met:
```python
import Note.DL.process.kernel as k   #import kernel
import tensorflow as tf
import nn as n                          #import neural network
from multiprocessing import Process,Lock,Manager
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
x_train=x_train.reshape([60000,784])
nn=n.nn()                                #create neural network object
nn.build()
kernel=k.kernel(nn)   #start kernel
kernel.stop=True
kernel.end_loss=0.7                           #stop condition
kernel.process=7      #7 processes to train
kernel.data_segment_flag=True
kernel.epoch=6                #epoch:6
kernel.batch=32            #batch:32
kernel.PO=2                    #use PO3
kernel.data(x_train,y_train)   #input train data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=[Lock(),Lock()]
g_lock=Lock()
for p in range(7):
	Process(target=kernel.train,args=(p,lock,g_lock)).start()
```
### PO3：
```python
import Note.DL.process.kernel as k   #import kernel
import tensorflow as tf
import nn as n                          #import neural network
from multiprocessing import Process,Lock,Manager
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
nn.build()
kernel=k.kernel(nn)   #start kernel
kernel.process=7     #7 processes to train
kernel.data_segment_flag=True
kernel.epoch=6                #epoch:6
kernel.batch=32            #batch:32
kernel.PO=3                    #use PO4
kernel.data(x_train,y_train)   #input train data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=Lock()
for p in range(7): 
	Process(target=kernel.train,args=(p,lock)).start()
kernel.update_nn_param()
kernel.test(x_train,y_train,32)
```
```python
import Note.DL.process.kernel as k   #import kernel
import tensorflow as tf
import nn as n                          #import neural network
from multiprocessing import Process,Lock,Manager
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
nn.build()
kernel=k.kernel(nn)   #start kernel
kernel.process=7     #7 processes to train
kernel.data_segment_flag=True
kernel.epoch=6                #epoch:6
kernel.batch=32            #batch:32
kernel.PO=3                    #use PO4
kernel.data(x_train,y_train)   #input train data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
for p in range(7): 
	Process(target=kernel.train,args=(p,)).start()
kernel.update_nn_param()
kernel.test(x_train,y_train,32)
```


## RL：
### Pool Network:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/pool%20net/DQN.py

#### PO2:
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
from multiprocessing import Process,Lock,Manager
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
manager=Manager()        #create manager object
kernel.init(manager)     #initialize shared data
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=2                    #use PO2
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()]
lock=[Lock(),Lock(),Lock()]
g_lock=Lock()
for p in range(5):
    Process(target=kernel.train,args=(p,100,lock,pool_lock,g_lock)).start()
```

#### PO3:
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
from multiprocessing import Process,Lock,Manager
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
manager=Manager()        #create manager object
kernel.init(manager)     #initialize shared data
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=3                    #use PO3
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()]
lock=[Lock(),Lock()]
g_lock=Lock()
for p in range(5):
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start()
```
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
from multiprocessing import Process,Lock,Manager
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
manager=Manager()        #create manager object
kernel.init(manager)     #initialize shared data
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=3                    #use PO3
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()]
lock=[Lock(),Lock(),Lock()]
g_lock=Lock()
for p in range(5):
    Process(target=kernel.train,args=(p,100,lock,pool_lock)).start()
```


# Online training:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn_ol.py

**example:**
```python
import Note.DL.kernel as k   #import kernel #import platform
import nn_ol as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn(x_train,y_train)                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train)   #input train data
kernel.train_ol()         #train neural network
```


# Test neural network:
## DL:
You can test it before using the kernel training neural network.
```python
import Note.DL.dl.test_nn as t
import tensorflow as tf              #import platform
import nn as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()
t.test(cnn,tf,x_train[:32],y_train[:32])
```

## RL:
You can test it before using the kernel training neural network.
```python
import Note.RL.rl.test_nn as t
import tensorflow as tf              #import platform
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
t.test(dqn,tf,2)
```


# Note Compiler:
documentation:https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/Note%207.0%20pv%20documentation/compiler
```python
import Note.nc as nc
c=nc.compiler('nn.n')
c.Compile()
```
