# Build agent:
You can learn how to build an agent from the examples in the following link.

https://github.com/NoteDance/Note/tree/Note-7.0/Note/neuralnetwork/docs_example

# Train:
```python
import tensorflow as tf
from Note.neuralnetwork.docs_example.DQN import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/neuralnetwork/docs_example/DQN.py
# from Note.neuralnetwork.docs_example.DQN_keras import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/neuralnetwork/docs_example/DQN_keras.py

model=DQN(4,128,2)
model.action_count=2
model.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.fit(train_loss, optimizer, 100)

# If set criterion.
# model.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200)
# model.fit(train_loss, optimizer, 100)

# If use prioritized replay.
# model.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200,pr=True,initial_TD=7,alpha=0.7)
# model.fit(train_loss, optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.fit(train_loss, optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.fit(train_loss, optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.fit(train_loss, optimizer, 100)

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
import torch
from Note.neuralnetwork.docs_example.DQN_pytorch import DQN # https://github.com/NoteDance/Note/blob/Note-7.0/Note/neuralnetwork/docs_example/DQN_pytorch.py

model=DQN(4,128,2)
model.action_count=2
model.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
optimizer = torch.optim.Adam(model.param)
model.fit(optimizer, 100)

# If set criterion.
# model.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200)
# model.fit(optimizer, 100)

# If use prioritized replay.
# model.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_count=10,criterion=200,pr=True,initial_TD=7,alpha=0.7)
# model.fit(optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.fit(optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.fit(optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.fit(optimizer, 100)

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
# Use HER
import tensorflow as tf
from Note.neuralnetwork.docs_example.DDPG_HER import DDPG # https://github.com/NoteDance/Note/blob/Note-7.0/Note/neuralnetwork/docs_example/DDPG_HER.py

model=DDPG(128,0.1,0.98,0.005)
model.set_up(pool_size=10000,batch=256,HER=True)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.fit(train_loss, optimizer, 10)
```
