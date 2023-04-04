# Stop training and saving when condition is met：
## DL：
**example(Stop training and saving when condition is met.):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)                 #start kernel
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
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
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

## RL：
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


# PO3：
**multithreading example(PO3):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.process_thread=7                        #thread count,use 7 threads to train
kernel.PO=3
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


# Segment data：
**multithreading example(segment data):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
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


# Parallel test：
**example(parallel test):**
```python
import Note.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn_acc as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)                 #start kernel
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
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
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
