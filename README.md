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
kernel.restrained_parallelism=True
kernel.process_thread=2                        #thread count,use 2 threads to train
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
kernel.multiprocessing_threading=threading
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


# PO3:
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
kernel.process_thread=7                        #thread count,use 7 threads to train
kernel.PO=3                    #use PO3
kernel.multiprocessing_threading=threading
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


# Segment data:
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
kernel.process_thread=7                        #thread count,use 7 threads to train
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
kernel.multiprocessing_threading=threading
kernel.process_thread_t=6                #test thread count
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train,x_test,y_test)   #input you data
kernel.train(32,5,32)         #train neural network
                           #batch size:32
			   #test batch size:32
                           #epoch:5
kernel.save()              #save neural network
```

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
kernel.multiprocessing_threading=threading
kernel.process_thread=7                        #thread count,use 7 threads to train
kernel.process_thread_t=6                      #test thread count
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


#  Online training:
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
kernel.process_thread=7                        #thread count,use 7 thread to train
kernel.PO=2
kernel.create_pt_num(7)   #input you data
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


# Layer:
**You can download neural network example in this link,and then you can import neural network and train with kernel,link and example code are below.**

https://github.com/NoteDancing/Note-documentation/blob/Note-7.0-pv/Note%207.0%20pv%20documentation/DL/neural%20network/tensorflow/nn_layer.py
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import nn_layer as n                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
x_train=x_train.reshape([60000,784])
nn=n.nn()                                #create neural network object
nn.build()                          #build neural network
kernel=k.kernel(nn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train)   #input you data
kernel.train(32,5)         #train neural network
```
