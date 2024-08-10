# Train:
Agent built with Note and Keras.
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/DQN.py
# from Note.models.docs_example.RL.keras.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/DQN.py

model=DQN(4,128,2)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)

# If set criterion.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(train_loss, optimizer, 100)

# If use prioritized replay.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.train(train_loss, optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(train_loss, optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(train_loss, optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(train_loss, optimizer, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()
# model.visualize_comparison()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use PPO
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.PPO import PPO # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/PPO.py
# from Note.models.docs_example.RL.keras.PPO import PPO # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/PPO.py

model=PPO(4,128,2,0.7,0.7)
model.set_up(policy=rl.SoftmaxPolicy(),pool_size=10000,batch=64,update_steps=1000,PPO=True)
optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
```
```python
# Use HER
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/DDPG_HER.py
# from Note.models.docs_example.RL.keras.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/DDPG_HER.py

model=DDPG(128,0.1,0.98,0.005)
model.set_up(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.MADDPG import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/MADDPG.py
# from Note.models.docs_example.RL.keras.MADDPG import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/MADDPG.py

model=DDPG(128,0.1,0.98,0.005)
model.set_up(noise=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DQN.py
import multiprocessing as mp

model=DQN(4,128,2,7)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
manager=mp.Manager()
model.train(train_loss, optimizer, 100, mp=mp, manager=manager, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DDPG_HER.py
import multiprocessing as mp

model=DDPG(128,0.1,0.98,0.005,7)
model.set_up(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
manager=mp.Manager()
model.train(train_loss, optimizer, 2000, mp=mp, manager=manager, processes=7, processes_her=4)
```
Agent built with PyTorch.
```python
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/pytorch/DQN_pytorch.py

model=DQN(4,128,2)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100)

# If set criterion.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(optimizer, 100)

# If use prioritized replay.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.train(optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(optimizer, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()
# model.visualize_comparison()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/pytorch/multiprocessing/DQN.py
import multiprocessing as mp

model=DQN(4,128,2,7)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = torch.optim.Adam(model.param)
manager=mp.Manager()
model.train(optimizer, 100, mp=mp, manager=manager, processes=7)
```

# Distributed training:
Agent built with Note and Keras.
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/DQN.py
# from Note.models.docs_example.RL.keras.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/DQN.py

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2)
  optimizer = tf.keras.optimizers.Adam()
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If set criterion.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If use prioritized replay.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()
# model.visualize_comparison()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DQN.py
import multiprocessing as mp

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2,7)
  optimizer = tf.keras.optimizers.Adam()
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
manager=mp.Manager()
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100, mp=mp, manager=manager, processes=7)
```
