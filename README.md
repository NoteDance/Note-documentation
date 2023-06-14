# Stop training and saving when condition is met:
## DL:
**example(Stop training and saving when condition is met.):**
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
kernel.data(x_train,y_train)   #input you data
kernel.train(32,5)         #train neural network
                           #batch size:32
                           #epoch:5
```

## RL:
**example(Stop training and saving when condition is met.):**
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

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn_acc.py

**example(parallel test):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn_acc as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.threading=threading
kernel.thread_t=6                #test thread count
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train,x_test,y_test)   #input you data
kernel.train(32,5,32)         #train neural network
                           #batch size:32
			   #test batch size:32
                           #epoch:5
kernel.save()              #save neural network
```

**example(test module):**
```python
import nn_acc as n
import Note.DL.dl.test as t
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()
test=t.parallel_test(nn,x_test,y_test,6,32)
test.segment_data()
class thread(threading.Thread):     
	def run(self):              
		test.test()
for _ in range(6):
	_thread=thread()
	_thread.start()
for _ in range(6):
	_thread.join()
loss,acc=test.loss_acc()
```


# Multiprocessing:
## PO3:
**multiprocessing example(PO3):**
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
kernel.PO=3                    #use PO3
kernel.data(x_train,y_train)   #input you data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=[Lock(),Lock()]
g_lock=[Lock(),Lock(),Lock()]
for p in range(7):
	Process(target=kernel.train,args=(p,lock,g_lock)).start()
kernel.update_nn_param()
kernel.test(x_train,y_train,32)
```
## Stop training and saving when condition is met:
**multiprocessing example(Stop training and saving when condition is met.):**
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
kernel.PO=3                    #use PO3
kernel.data(x_train,y_train)   #input you data
manager=Manager()              #create manager object
kernel.init(manager)      #initialize shared data
lock=[Lock(),Lock()]
g_lock=[Lock(),Lock(),Lock()]
for p in range(7):
	Process(target=kernel.train,args=(p,lock,g_lock)).start()
```


# Multithreading:
## DL：
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn.py

**multithreading example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=7                        #thread count,use 7 threads to train
kernel.epoch_=6                #epoch:6
kernel.PO=2                    #use PO2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32) #batch size:32
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
kernel.visualize_train()
```

**multithreading example(segment data):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=7                        #thread count,use 7 threads to train
kernel.data_segment_flag=True
kernel.batches=1875            #batches:1875
kernel.epoch_=6                #epoch:6
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32) #batch size:32
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
kernel.visualize_train()
```

**multithreading example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=2                        #thread count,use 2 threads to train
kernel.PO=2                    #use PO2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):     
	def run(self):              
		kernel.train(32,3)  #batch size:32 epoch:3
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.visualize_train()
```

### Stop training and saving when condition is met:
**multithreading example(Stop training and saving when condition is met.):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.stop=True
kernel.end_loss=0.7                           #stop condition
kernel.thread=2                        #thread count,use 2 threads to train
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3) #batch size:32 epoch:3
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```

**Gradient Attenuation：**

**Calculate the attenuation coefficient based on the optimization counter using the attenuation function.**

**example:https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn_attenuate.py**

## RL：
### Pool Network:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/RL/neural%20network/tensorflow/pool%20net/thread/DQN.py

**multithreading example:**
```python
import Note.RL.thread.kernel as k   #import kernel
import DQN as d
import threading
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=1                    #use PO1
kernel.threading=threading
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(5):
	_thread=thread()
	_thread.start()
for _ in range(5):
	_thread.join()
kernel.visualize_train()
kernel.visualize_reward()
```

### Stop training and saving when condition is met:
**multithreading example(Stop training and saving when condition is met.):**
```python
import Note.RL.kernel as k   #import kernel
import DQN as d
import threading
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn,5)   #start kernel,use 5 thread to train
kernel.stop=True
kernel.action_count=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200)
kernel.PO=1
kernel.threading=threading
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(5):
	_thread=thread()
	_thread.start()
for _ in range(5):
	_thread.join()
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

## PO3:
**multithreading example(PO3):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=7                        #thread count,use 7 threads to train
kernel.PO=3                    #use PO3
kernel.threading=threading
kernel.max_lock=7
kernel.data(x_train,y_train)   #input you data
kernel.lock=[threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,1) #batch size:32 epoch:1
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
```

## Parallel test:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn_acc.py

**multithreading example(parallel test):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn_acc as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.threading=threading
kernel.thread=7                        #thread count,use 7 threads to train
kernel.thread_t=6                      #test thread count
kernel.epoch_=6                #epoch:6
kernel.PO=2
kernel.data(x_train,y_train,x_test,y_test)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,test_batch=32) #batch size:32
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
kernel.visualize_train()
```

## Online training:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn_ol_p.py

**multithreading example:**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn_ol_p as n                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
nn=n.nn(x_train,y_train)                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=7                        #thread count,use 7 thread to train
kernel.PO=2
kernel.create_t_num(7)   #input you data
kernel.lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train_ol()
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
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
kernel.data(x_train,y_train)   #input you data
kernel.train_ol()         #train neural network
```


# Note Compiler:
documentation:https://github.com/NoteDancing/Note-documentation/tree/Note-7.0-pv/Note%207.0%20pv%20documentation/compiler
```python
import Note.nc as nc
c=nc.compiler('nn.n')
c.Compile()
```
