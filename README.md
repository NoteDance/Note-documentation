# Train:
## Note and Keras:
Agent built with Note or Keras.
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
model.set_up(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DQN.py
# from Note.models.docs_example.RL.keras.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/multiprocessing/DQN.py

model=DQN(4,128,2,7)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=True, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DDPG_HER.py

model=DDPG(128,0.1,0.98,0.005,7)
model.set_up(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000, pool_network=True, processes=7, processes_her=4)
```
## PyTorch:
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

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use HER
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/pytorch/DDPG_HER.py

model=DDPG(128,0.1,0.98,0.005)
model.set_up(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.MADDPG import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/pytorch/MADDPG.py

model=DDPG(128,0.1,0.98,0.005)
model.set_up(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/pytorch/multiprocessing/DQN.py

model=DQN(4,128,2,7)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100, pool_network=True, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.multiprocessing.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/pytorch/multiprocessing/DDPG_HER.py

model=DDPG(128,0.1,0.98,0.005,7)
model.set_up(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(train_loss, optimizer, 2000, pool_network=True, processes=7, processes_her=4)
```

# Distributed training:
Agent built with Note or Keras.
## MirroredStrategy:
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
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If set criterion.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If use prioritized replay.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

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

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=PPO(4,128,2,0.7,0.7)
  optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]

model.set_up(policy=rl.SoftmaxPolicy(),pool_size=10000,update_steps=1000,PPO=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)
```
```python
# Use HER
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/DDPG_HER.py
# from Note.models.docs_example.RL.keras.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/DDPG_HER.py

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set_up(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,criterion=-5,trial_count=10,HER=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 2000)
```
```python
# Use Multi-agent reinforcement learning
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.MADDPG import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/MADDPG.py
# from Note.models.docs_example.RL.keras.MADDPG import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/MADDPG.py

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set_up(policy=rl.SoftmaxPolicy(),pool_size=3000,trial_count=10,MA=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DQN.py
# from Note.models.docs_example.RL.keras.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/multiprocessing/DQN.py

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2,7)
  optimizer = tf.keras.optimizers.Adam()
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100, pool_network=True, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DDPG_HER.py

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005,7)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
model.set_up(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,trial_count=10,HER=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 2000, pool_network=True, processes=7, processes_her=4)
```
## MultiWorkerMirroredStrategy:
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/multiprocessing/DQN.py
# from Note.models.docs_example.RL.keras.multiprocessing.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/multiprocessing/DQN.py
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

strategy = tf.distribute.MultiWorkerMirroredStrategy()
per_worker_batch_size = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers

with strategy.scope():
  multi_worker_model = DQN(4,128,2)
  optimizer = tf.keras.optimizers.Adam()

multi_worker_model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
                    pool_network=True, processes=7)

# If set criterion.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If use prioritized replay.
# model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```

# Experimental:
## Note:
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.experimental.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/experimental/DQN.py

model=DQN(4,128,2,7)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=True, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.experimental.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/experimental/DQN.py

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2,7)
  optimizer = tf.keras.optimizers.Adam()
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100, pool_network=True, processes=7)
```
## PyTorch:
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.experimental.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/pytorch/experimental/DQN.py

model=DQN(4,128,2,7)
model.set_up(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100, pool_network=True, processes=7)
```
