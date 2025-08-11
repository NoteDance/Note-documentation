# Train:
**Note and Keras:**

Agent built with Note or Keras.
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DQN import DQN
# from Note.models.docs_example.RL.keras.DQN import DQN

model=DQN(4,128,2)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=False)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(train_loss, optimizer, 100, pool_network=False)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(train_loss, optimizer, 100, pool_network=False)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(train_loss, optimizer, 100, pool_network=False)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(train_loss, optimizer, 100, pool_network=False)

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
# Use PPO.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.PPO import PPO
# from Note.models.docs_example.RL.keras.PPO import PPO

model=PPO(4,128,2,0.7,0.7)
model.set(policy=rl.SoftmaxPolicy(),pool_size=10000,batch=64,update_steps=1000,PPO=True)
optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=False)
```
```python
# Use HER.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DDPG_HER import DDPG
# from Note.models.docs_example.RL.keras.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000, pool_network=False)
```
```python
# Use Multi-agent reinforcement learning.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.MADDPG import DDPG
# from Note.models.docs_example.RL.keras.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MARL=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=False)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN import DQN

model=DQN(4,128,2,7)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=True, processes=7)
```
```python
# Use HER.
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005,7)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000, pool_network=True, processes=7, processes_her=4)
```
```python
# Use prioritized replay.
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN_PR import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN_PR import DQN

model=DQN(4,128,2,7)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=True, processes=7, processes_pr=4)
```
**PyTorch:**

Agent built with PyTorch.
```python
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DQN import DQN

model=DQN(4,128,2)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100, pool_network=False)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(optimizer, 100, pool_network=False)

# If use prioritized replay.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.train(optimizer, 100, pool_network=False)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(optimizer, 100, pool_network=False)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(optimizer, 100, pool_network=False)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(optimizer, 100, pool_network=False)

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
# Use HER.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 2000, pool_network=False)
```
```python
# Use Multi-agent reinforcement learning.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MARL=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 100, pool_network=False)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.pool_network.DQN import DQN

model=DQN(4,128,2,7)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100, pool_network=True, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.pool_network.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005,7)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(train_loss, optimizer, 2000, pool_network=True, processes=7, processes_her=4)
```

# Distributed training:
Agent built with Note or Keras.
**MirroredStrategy:**
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DQN import DQN
# from Note.models.docs_example.RL.keras.DQN import DQN

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2)
  optimizer = tf.keras.optimizers.Adam()
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=GLOBAL_BATCH_SIZE,update_steps=10)
model.distributed_training(optimizer, strategy, 100, pool_network=False)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=GLOBAL_BATCH_SIZE,update_steps=10,trial_count=10,criterion=200)
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

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
# Use PPO.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.PPO import PPO
# from Note.models.docs_example.RL.keras.PPO import PPO

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=PPO(4,128,2,0.7,0.7)
  optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]

model.set(policy=rl.SoftmaxPolicy(),pool_size=10000,batch=GLOBAL_BATCH_SIZE,update_steps=1000,PPO=True)
model.distributed_training(optimizer, strategy, 100, pool_network=False)
```
```python
# Use HER.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.DDPG_HER import DDPG
# from Note.models.docs_example.RL.keras.DDPG_HER import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=GLOBAL_BATCH_SIZE,criterion=-5,trial_count=10,HER=True)
model.distributed_training(optimizer, strategy, 2000, pool_network=False)
```
```python
# Use Multi-agent reinforcement learning
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.MADDPG import DDPG
# from Note.models.docs_example.RL.keras.MADDPG import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,trial_count=10,MARL=True)
model.distributed_training(optimizer, strategy, 100, pool_network=False)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN import DQN

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2,7)
  optimizer = tf.keras.optimizers.Adam()
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=GLOBAL_BATCH_SIZE,update_batches=17)
model.distributed_training(optimizer, strategy, 100, pool_network=True, processes=7)
```
```python
# Use HER.
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DDPG_HER import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005,7)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=GLOBAL_BATCH_SIZE,trial_count=10,HER=True)
model.distributed_training(optimizer, strategy, 2000, pool_network=True, processes=7, processes_her=4)
```
**MultiWorkerMirroredStrategy:**
```python
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.pool_network.DQN import DQN
# from Note.models.docs_example.RL.keras.pool_network.DQN import DQN
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

multi_worker_model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=global_batch_size,update_batches=17)
multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
                    pool_network=True, processes=7)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=global_batch_size,update_steps=10,trial_count=10,criterion=200)
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
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

# RL.set:

**Description**:
The `set` function configures various parameters of the Reinforcement Learning (RL) agent. These parameters control the policy, noise, experience pool, batch size, update frequency, and training termination conditions. By adjusting these settings, users can fine-tune the agent's behavior and training process to suit specific RL tasks.

---

**Function Signature**:
```python
def set(self, 
        policy=None, 
        noise=None, 
        pool_size=None, 
        batch=None, 
        update_batches=None, 
        update_steps=None, 
        trial_count=None, 
        criterion=None, 
        PPO=False, 
        HER=False, 
        MARL=False, 
        PR=False, 
        epsilon=None, 
        initial_TD=7., 
        alpha=0.7):
```

---

**Parameter Description**:

- **`policy`** (`rl.Policy` or `None`):  
  Specifies the policy object for the agent, which controls how actions are selected in each state.

- **`noise`** (`float` or `None`):  
  Adds noise to the action selection process, typically used in continuous action spaces to encourage exploration. The default value is `None`.

- **`pool_size`** (`int` or `None`):  
  The size of the experience replay pool, i.e., the maximum number of experiences that can be stored in the pool. If not set, the pool size defaults to the internal value.

- **`batch`** (`int` or `None`):  
  The size of the batch sampled from the experience pool during training, affecting how much data is used in each training step.

- **`update_batches`** (`int` or `None`):  
  The number of batches to use when updating the network, applicable when using a pool network for storing experiences. Defaults to `None`.

- **`update_steps`** (`int` or `None`):  
  The frequency at which the target Q network is updated, in terms of the number of steps.

- **`trial_count`** (`int` or `None`):  
  Specifies the number of trials over which the average reward is computed during training. After every `trial_count` episodes, the agent's performance is evaluated by calculating the average reward over those episodes. If not set, no average reward is calculated.

- **`criterion`** (`float` or `None`):  
  Specifies the threshold used to terminate training. If `trial_count` is set, the average reward over the most recent `trial_count` episodes is calculated. If this average reward meets or exceeds `criterion`, training is terminated early. This helps avoid unnecessary training once the desired performance level is achieved.

- **`PPO`** (`bool`):  
  Whether to use the Proximal Policy Optimization (PPO) algorithm.

- **`HER`** (`bool`):  
  Whether to use Hindsight Experience Replay (HER), typically used for goal-oriented tasks.

- **`MARL`** (`bool`):  
  Whether to use Multi-Agent (MARL) reinforcement learning.

- **`PR`** (`bool`):  
  Whether to use Prioritized Experience Replay (PR), a technique to sample experiences based on their significance.
  
- **`IRL`** (`bool`):  
  Whether to use Inverse Reinforcement Learning (IRL) to estimate the reward function based on expert trajectories. Setting this to `True` enables IRL functionality.

- **`epsilon`** (`float` or `None`):  
  The `ε` value used in an `ε-greedy` policy, controlling the probability of choosing a random action to encourage exploration.

- **`initial_TD`** (`float`):  
  The initial TD-error value used in Prioritized Replay. A higher TD-error leads to higher prioritization of the sample in the experience pool.

- **`alpha`** (`float`):  
  The `α` value used in Prioritized Replay, determining how much the TD-error influences sample prioritization. A higher `α` increases the importance of prioritizing higher TD-error experiences.

---

**Usage Example**:

```python
# Create an instance of a DQN agent
model = DQN(state_dim=4, hidden_dim=128, action_dim=2)

# Set the agent's policy, experience pool size, batch size, and early stopping conditions
model.set(
    policy=rl.EpsGreedyQPolicy(epsilon=0.01),  # Use epsilon-greedy policy
    pool_size=10000,                           # Set experience pool size
    batch=64,                                  # Set batch size
    update_steps=10,                           # Update target network every 10 steps
    trial_count=100,                           # Calculate average reward every 100 trials
    criterion=200.0,                           # Stop training if average reward reaches 200
    PR=True,                                   # Enable Prioritized Replay
    initial_TD=7.0,                            # Initial TD-error set to 7.0
    alpha=0.7                                  # Alpha value for prioritized sampling
)
```

In this example, the agent computes the average reward every 100 trials. If the average reward reaches 200 or higher, the training process stops early. This method allows the agent to stop training once it reaches a desired performance level, improving training efficiency.

# RL.train:
**Description**:
This function handles the training loop of the reinforcement learning (RL) agent. It supports both single-process and multi-process training, along with the option to use a **pool network** for experience replay. Additionally, it provides support for Hindsight Experience Replay (HER), Prioritized Experience Replay (PR), and optional just-in-time (JIT) compilation for performance optimization.

**Arguments**:

- **`train_loss`** (`tf.keras.metrics.Metric`): The loss metric used to evaluate the training loss during the optimization process.
  
- **`optimizer`** (`tf.keras.optimizers.Optimizer`): The optimizer used to update the model parameters during training.
  
- **`episodes`** (`int`, optional): The number of training episodes to run. If `None`, the training will continue indefinitely until a stopping criterion is met.

- **`jit_compile`** (`bool`, optional, default=`True`): Whether to enable TensorFlow's JIT compilation for improved performance during training.
  
- **`pool_network`** (`bool`, optional, default=`True`): Whether to use a pool network for experiences collection.
  
- **`processes`** (`int`, optional): Number of parallel processes to use for data collection when using a pool network. If `None`, multi-processing is disabled.
  
- **`processes_her`** (`int`, optional): Number of parallel processes dedicated to Hindsight Experience Replay (HER). Only used if HER is enabled.
  
- **`processes_pr`** (`int`, optional): Number of parallel processes dedicated to Prioritized Experience Replay (PR). Only used if PR is enabled.
  
- **`shuffle`** (`bool`, optional, default=`False`): If `True`, experiences in the pool will be shuffled before sampling. This can help prevent overfitting to recent experiences.
  
- **`p`** (`int`, optional): A parameter that determines the update frequency for logging and printing intermediate results. If `None`, it defaults to `9`.

**Returns**:
- No return value. The function prints progress at specified intervals and updates the model's parameters based on the training procedure.

**Details**:
1. **Multiprocessing Setup**:
   - If `pool_network=True`, the function sets up parallel processes to collect experiences in parallel using Python's `multiprocessing` library. Each process collects states, actions, rewards, and other necessary information, which are then aggregated into a shared experience pool.
   
2. **Training Procedure**:
   - If a pool network is used, the agent gathers experiences from multiple parallel environments or processes and stores them in a shared memory pool. The training loop then samples batches from this pool to update the agent's neural network. Otherwise, the agent igathers experiences from environment and stores them in a pool and then updates the network using a different training method (`train2`).
   
   - For each episode, the loss is computed and accumulated in `self.loss_list`. This loss represents the agent's learning progress, and the model parameters are updated using the provided optimizer.
   
3. **Handling Special Experience Replay**:
   - **Hindsight Experience Replay (HER)**: If HER is enabled, the function creates additional processes to manage HER-specific experience sampling and updates.
   
   - **Prioritized Experience Replay (PR)**: If PR is enabled, a prioritized experience replay buffer is updated with the TD-errors (Temporal Difference) of the experiences.

4. **Logging and Saving**:
   - The function prints progress messages every `p` episodes and logs key metrics like average reward and loss. The model can be saved at regular intervals (`self.save_freq`) and upon achieving a certain reward criterion (`self.criterion`).

5. **Termination Criteria**:
   - Training continues until the specified number of episodes (`episodes`) is reached, or in infinite mode (when `episodes=None`), until the reward criterion is met.

**Usage Example**:

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Start training for 100 episodes using a pool network with 8 processes
agent.train(train_loss=train_loss, optimizer=optimizer, episodes=100, pool_network=True, processes=8)
```

This documentation provides a detailed explanation of each parameter and the internal behavior of the function, which should be useful for understanding its usage in reinforcement learning training loops.

# RL.distributed_training:

**Description**:
The `distributed_training` function is designed to handle distributed reinforcement learning (RL) training across multiple devices or workers. It supports various TensorFlow strategies, such as MirroredStrategy, MultiWorkerMirroredStrategy, and ParameterServerStrategy. The function is optimized for both single-node and multi-node setups, enabling distributed training with optional experience replay buffers, including prioritized and hindsight experience replay (HER). 

This function also supports parallel data collection through a **pool network** and optional just-in-time (JIT) compilation for performance optimization.

**Parameters**:

- **`optimizer`** (`tf.keras.optimizers.Optimizer`): The optimizer used to update model parameters during training.

- **`strategy`** (`tf.distribute.Strategy`): A TensorFlow distribution strategy to manage the distributed training setup. This could be `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `ParameterServerStrategy`.

- **`episodes`** (`int`, optional): The number of training episodes to run. If set to `None`, the function will run indefinitely.

- **`num_episodes`** (`int`, optional): Alternative to `episodes`, used in specific strategy cases like `MultiWorkerMirroredStrategy`. Defaults to `None`.

- **`jit_compile`** (`bool`, optional, default=`True`): Whether to enable TensorFlow's Just-In-Time (JIT) compilation for performance optimization.

- **`pool_network`** (`bool`, optional, default=`True`): Whether to use a pool network for experiences collection.

- **`processes`** (`int`, optional): The number of parallel processes to use for data collection when `pool_network` is enabled. If set to `None`, multiprocessing is disabled.

- **`processes_her`** (`int`, optional): The number of parallel processes dedicated to Hindsight Experience Replay (HER) data collection, if HER is enabled.

- **`processes_pr`** (`int`, optional): The number of parallel processes for prioritized experience replay (PR) data collection, if PR is enabled.

- **`shuffle`** (`bool`, optional, default=`False`): If `True`, shuffles the data in the pool before training to prevent overfitting to recent experiences.

- **`p`** (`int`, optional): Controls how frequently to log intermediate results. If set to `None`, it defaults to `p=9`.

**Returns**:
- **None**. The function logs training progress, including loss and reward information, at specified intervals. It may also save model parameters based on a given frequency.

**Details**:

1. **Training with Distribution Strategies**:
   - The function adapts to various TensorFlow distribution strategies:
     - **`MirroredStrategy`**: For synchronous training across multiple GPUs on a single machine.
     - **`MultiWorkerMirroredStrategy`**: For synchronous training across multiple workers.
     - **`ParameterServerStrategy`**: For asynchronous training with parameter servers.

2. **Parallel Data Collection (Pool Network)**:
   - When `pool_network` is enabled, the function sets up parallel processes using Python's `multiprocessing` to collect experience (state, action, reward, next-state, done) from multiple environments. The data is stored in shared memory using multiprocessing managers.
   - The data can be used to update the agent’s neural network either through traditional replay or advanced methods like HER or prioritized replay.

3. **Handling HER and PR**:
   - If HER is enabled (`processes_her` is not `None`), the function initializes additional buffers and processes to handle HER-specific data collection.
   - Similarly, for prioritized replay (`processes_pr` is not `None`), the function maintains a TD-error (temporal difference error) list to prioritize experiences during replay.

4. **Training Execution**:
   - For each episode, the function collects experience using the pool network (if enabled) and updates the agent’s model parameters through the specified optimizer and distribution strategy. The loss is calculated either through a customized `train1` method (pool network) or `train2` method (direct training).
   - After every few episodes (controlled by `p`), the function logs the loss, reward, and progress. If a performance criterion is met (e.g., a certain average reward threshold), the training may terminate early.

5. **Model Saving**:
   - The function saves model parameters periodically, based on a pre-specified frequency (`save_freq`). If the parameter `save_param_only` is set, only model parameters are saved, otherwise the full model is saved.

6. **Time Tracking**:
   - The function keeps track of the total training time, logging it at the end of the training session.

**Usage Example**:

```python
# Example usage of the distributed_training function
global_batch_size = 64
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
strategy = tf.distribute.MirroredStrategy()

agent.distributed_training(
    global_batch_size=global_batch_size, 
    optimizer=optimizer, 
    strategy=strategy, 
    episodes=100, 
    pool_network=True, 
    processes=8
)
```

In this example, the function runs distributed training using the `MirroredStrategy`, where experience is collected in parallel through 8 processes and stored in a pool buffer. Training runs for 100 episodes with a global batch size of 64.

# RL.adjust_window_size:

**Description**:
Compute an adaptive experience-replay window size based on the *effective sample size* (ESS) of the prioritized weights. This function estimates how many recent experiences should be kept (vs. discarded) by converting the ESS into a desired number of kept samples, applying optional exponential moving average (EMA) smoothing to the ESS, and returning the number of oldest entries to drop (the window size). It supports both single-process and pool-network (multi-process) setups.

**Arguments**:

* **`p`** (`int`): Process index when `pool_network=True`. Selects which sub-pool's weight vector to evaluate. If `pool_network=False`, `p` is ignored.
* **`scale`** (`float`, optional, default=`1.0`): Multiplier applied to the (smoothed) ESS to compute the desired number of samples to keep. Values >1 increase the kept size (smaller window), values <1 decrease it (larger window).
* **`smooth_alpha`** (`float`, optional, default=`0.2`): EMA smoothing coefficient in `[0,1]` used to smooth ESS over time. Higher values weight the newest ESS more; lower values emphasize past ESS.

**Returns**:

* **`window_size`** (`int`): Suggested number of oldest samples to remove from the experience pool. Computed as `len(weights) - desired_keep`. Guaranteed to be a non-negative integer under normal conditions (see Notes).

**Details**:

1. **Choose source of weights**:

   * If `self.pool_network == True` the function reads `weights = np.array(self.ratio_list[p])` — the per-process ratio array used for prioritized sampling in that sub-pool.
   * If `self.pool_network == False` the function reads `weights = np.array(self.prioritized_replay.ratio)` — the global prioritized weights array.

2. **Compute ESS**:

   * Calls `self.compute_ess_from_weights(weights)`, which:

     * clips weights to a minimum positive value (to avoid zeros),
     * normalizes them to a probability vector `p`,
     * computes ESS as `1 / sum(p^2)`.
   * ESS is a continuous estimate of how many “effective” independent samples exist given the weight distribution.

3. **EMA smoothing**:

   * The function stores smoothed ESS in `self.ema_ess`.
   * For `pool_network==True`, `self.ema_ess` is a list and `self.ema_ess[p]` is updated. For single-process mode it is a scalar.
   * New smoothed ESS is `ema = smooth_alpha * ess + (1.0 - smooth_alpha) * prev_ema` (or `ema = ess` if no prior EMA exists).

4. **Desired kept samples and window size**:

   * `desired_keep = np.clip(int(ema * scale), 1, len(weights) - 1)`

     * Intuition: convert (smoothed) ESS to an integer number of samples to keep, optionally scaled.
     * The clip prevents degeneracy by requiring at least one sample kept and at most `len(weights)-1`.
   * `window_size = len(weights) - desired_keep`

     * This is the number of oldest entries to remove; the caller can then slice arrays like `state_pool = state_pool[window_size:]`.

5. **Side effects**:

   * Updates `self.ema_ess` (or `self.ema_ess[p]`) with the new smoothed ESS value.
   * Does **not** modify replay buffers or ratio/TD arrays — it only returns the window size. The caller is responsible for actually removing entries.

6. **Assumptions & edge cases**:

   * The function assumes `weights` has length ≥ 2. If `len(weights) <= 1` the code `np.clip(..., 1, len(weights)-1)` may produce an invalid clip range (upper < lower) and raise a `ValueError` or produce unexpected results. It is recommended to guard against this by checking `len(weights)` before calling (or adding a small wrapper).
   * If weights contain zeros or extremely small values, `compute_ess_from_weights` already protects against divide-by-zero by clipping to a small positive minimum.

7. **Complexity**:

   * Time complexity is O(n) where n is the number of weights (dominant cost is computing ESS).

**Usage Example**:

https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/keras/PPO_pr.py
https://github.com/NoteDance/Note/blob/Note-7.0/Note/models/docs_example/RL/note/PPO_pr.py

# Policy classes:

**SoftmaxPolicy**

**Description**:
Implements a softmax policy for multinomial distribution. This policy selects actions based on a probability distribution, where each action has a probability proportional to the exponentiated Q-value or reward estimate.

**Methods**:

- **`select_action(nb_actions, probs)`**:
  - **Arguments**:
    - `nb_actions` (`int`): Number of possible actions.
    - `probs` (`np.ndarray`): A probability distribution over the actions.
  - **Returns**:
    - `action` (`int`): The action selected according to the probability distribution `probs`.

**Usage**:
Use `SoftmaxPolicy` with the `set` function of an RL agent:
```python
policy = SoftmaxPolicy()
model.set(policy=policy)
```

---

**EpsGreedyQPolicy**

**Description**:
Implements the epsilon-greedy policy. With probability `epsilon`, this policy selects a random action to encourage exploration. Otherwise, it selects the action with the highest estimated Q-value.

**Constructor**:

- **`__init__(eps=.1)`**:
  - **Arguments**:
    - `eps` (`float`): Epsilon value representing the probability of choosing a random action. Defaults to `0.1`.

**Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action, either random (with probability `eps`) or the best action (with probability `1 - eps`).

**Usage**:
Use `EpsGreedyQPolicy` with the `set` function of an RL agent:
```python
policy = EpsGreedyQPolicy(eps=0.05)
model.set(policy=policy)
```

---

**AdaptiveEpsGreedyPolicy**

**Description**:
Implements an adaptive epsilon-greedy policy. This policy dynamically adjusts the epsilon value based on the training step to balance exploration and exploitation. With a probability of `epsilon`, the policy selects a random action to encourage exploration; otherwise, it selects the action with the highest estimated Q-value.

**Constructor**:

- **`__init__(initial_eps=1.0, min_eps=0.1, decay_rate=0.0001)`**:
  - **Arguments**:
    - `initial_eps` (`float`): The initial exploration rate (epsilon). Defaults to `1.0`.
    - `min_eps` (`float`): The minimum epsilon value, representing the lowest exploration rate. Defaults to `0.1`.
    - `decay_rate` (`float`): The rate at which epsilon decreases over time. Defaults to `0.0001`.

**Methods**:

- **`select_action(q_values, step_counter)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
    - `step_counter` (`int`): The current training step, used to calculate the current epsilon value.
  - **Returns**:
    - `action` (`int`): The selected action, either random (with probability `epsilon`) or the best action (with probability `1 - epsilon`).

**Usage**:
Use `AdaptiveEpsGreedyPolicy` with the `set` function of an RL agent:
```python
policy = AdaptiveEpsGreedyPolicy(initial_eps=1.0, min_eps=0.1, decay_rate=0.0001)
model.set(policy=policy)
``` 

---

**GreedyQPolicy**

**Description**:
Implements the greedy policy, where the agent always selects the action with the highest estimated Q-value. This policy does not explore other actions.

**Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The action with the highest Q-value.

**Usage**:
Use `GreedyQPolicy` with the `set` function of an RL agent:
```python
policy = GreedyQPolicy()
model.set(policy=policy)
```

---

**BoltzmannQPolicy**

**Description**:
Implements the Boltzmann Q Policy. This policy selects actions based on a probability distribution derived from exponentiated Q-values, where higher Q-values have higher probabilities. The `tau` parameter controls the exploration: higher `tau` values result in more exploration, while lower values focus on exploitation.

**Constructor**:

- **`__init__(tau=1., clip=(-500., 500.))`**:
  - **Arguments**:
    - `tau` (`float`): Temperature parameter controlling exploration. Default is `1.0`.
    - `clip` (`tuple`): Range to clip the Q-values before exponentiation. Default is `(-500., 500.)`.

**Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action according to the Boltzmann distribution of Q-values.

**Usage**:
Use `BoltzmannQPolicy` with the `set` function of an RL agent:
```python
policy = BoltzmannQPolicy(tau=0.5)
model.set(policy=policy)
```

---

**MaxBoltzmannQPolicy**

**Description**:
Combines epsilon-greedy and Boltzmann Q-policy. With probability `epsilon`, the agent follows the Boltzmann distribution to select an action. With probability `1 - epsilon`, it selects the action with the highest Q-value.

**Constructor**:

- **`__init__(eps=.1, tau=1., clip=(-500., 500.))`**:
  - **Arguments**:
    - `eps` (`float`): Epsilon value for selecting random actions.
    - `tau` (`float`): Temperature parameter for Boltzmann exploration.
    - `clip` (`tuple`): Range to clip the Q-values. Default is `(-500., 500.)`.

**Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action, either based on Boltzmann exploration or the greedy choice.

**Usage**:
Use `MaxBoltzmannQPolicy` with the `set` function of an RL agent:
```python
policy = MaxBoltzmannQPolicy(eps=0.1, tau=0.5)
model.set(policy=policy)
```

---

**BoltzmannGumbelQPolicy**

**Description**:
Implements the Boltzmann-Gumbel exploration policy, which is invariant to the mean of rewards but sensitive to reward variance. This policy uses Gumbel noise to perturb the Q-values for exploration and adapts over time based on the parameter `C`.

**Constructor**:

- **`__init__(C=1.0)`**:
  - **Arguments**:
    - `C` (`float`): Exploration parameter to adjust for variance in rewards.

**Methods**:

- **`select_action(q_values, step_counter)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
    - `step_counter` (`int`): Current step of the training process.
  - **Returns**:
    - `action` (`int`): The selected action based on Boltzmann-Gumbel exploration.

**Usage**:
Use `BoltzmannGumbelQPolicy` with the `set` function of an RL agent:
```python
policy = BoltzmannGumbelQPolicy(C=1.0)
model.set(policy=policy)
```

---

**GumbelSoftmaxPolicy**

**Description**:
Implements the Gumbel Softmax policy for continuous action spaces. This policy samples from a Gumbel distribution and returns one-hot encoded actions for discrete action selection.

**Constructor**:

- **`__init__(temperature=1.0, eps=0.01)`**:
  - **Arguments**:
    - `temperature` (`float`): Temperature parameter for Gumbel sampling.
    - `eps` (`float`): Epsilon value for exploration in the one-hot encoding process.

**Methods**:

- **`onehot_from_logits(logits)`**:
  - **Arguments**:
    - `logits` (`np.ndarray`): The unnormalized log-probabilities (logits) for each action.
  - **Returns**:
    - `onehot_action` (`np.ndarray`): One-hot encoded action.

- **`sample_gumbel(shape, eps=1e-20)`**:
  - **Arguments**:
    - `shape` (`tuple`): Shape of the Gumbel sample to be drawn.
    - `eps` (`float`): Small epsilon to avoid numerical issues.
  - **Returns**:
    - `sample` (`np.ndarray`): Gumbel-distributed sample.

- **`gumbel_softmax_sample(logits)`**:
  - **Arguments**:
    - `logits` (`np.ndarray`): Logits for each action.
  - **Returns**:
    - `softmax_probs` (`np.ndarray`): Softmax probabilities for each action.

- **`gumbel_softmax(logits)`**:
  - **Arguments**:
    - `logits` (`np.ndarray`): Logits for each action.
  - **Returns**:
    - `y` (`np.ndarray`): One-hot encoded action sampled using Gumbel softmax.

**Usage**:
Use `GumbelSoftmaxPolicy` with the `set` function of an RL agent:
```python
policy = GumbelSoftmaxPolicy(temperature=0.5, eps=0.01)
model.set(policy=policy)
```

---

This documentation provides detailed descriptions and usage examples for each policy class and reflects the typical way of passing these policies to an RL agent using the `set` function.

# Noise classes:

**GaussianWhiteNoiseProcess**

**Description**:
Implements a Gaussian white noise process, generating noise from a Gaussian distribution with mean `mu` and time-varying standard deviation `sigma` (which anneals over time). This type of noise is commonly used in exploration strategies for continuous action spaces.

**Constructor**:
- **`__init__(mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1)`**:
  - **Arguments**:
    - `mu` (`float`): Mean of the Gaussian distribution.
    - `sigma` (`float`): Initial standard deviation.
    - `sigma_min` (`float`): Minimum standard deviation after annealing. If `None`, annealing is disabled.
    - `n_steps_annealing` (`int`): Number of steps over which the annealing occurs.
    - `size` (`int`): Size of the noise vector to be sampled.

**Methods**:

- **`sample()`**:
  - Generates a sample of Gaussian noise based on the current standard deviation (`sigma`), which anneals over time.
  - **Returns**:
    - `sample` (`np.ndarray`): Sampled noise from the Gaussian distribution.

**Usage**:
```python
noise = GaussianWhiteNoiseProcess(mu=0., sigma=1., sigma_min=0.1, n_steps_annealing=1000, size=1)
model.set(noise=noise)
```

---

**OrnsteinUhlenbeckProcess**

**Description**:
This process generates noise using the Ornstein-Uhlenbeck process, a continuous-time stochastic process often used to model time-correlated noise. It is frequently applied in reinforcement learning, especially for exploration in environments with continuous action spaces (e.g., DDPG). The noise tends to revert to the mean over time, controlled by the parameter `theta`.

**Constructor**:
- **`__init__(theta, mu=0., sigma=1., dt=1e-2, size=1, sigma_min=None, n_steps_annealing=1000)`**:
  - **Arguments**:
    - `theta` (`float`): Rate of mean reversion (higher `theta` means stronger pull towards `mu`).
    - `mu` (`float`): Mean value to which the process reverts.
    - `sigma` (`float`): Initial standard deviation of the noise.
    - `dt` (`float`): Time step for discretization.
    - `size` (`int`): Size of the noise vector.
    - `sigma_min` (`float`): Minimum standard deviation for annealing.
    - `n_steps_annealing` (`int`): Number of steps over which the standard deviation anneals.

**Methods**:

- **`sample()`**:
  - Generates a noise sample based on the current state of the process and updates the internal state.
  - **Returns**:
    - `x` (`np.ndarray`): Sampled noise from the Ornstein-Uhlenbeck process.

- **`reset_states()`**:
  - Resets the internal state (`x_prev`) of the process to a random value drawn from a Gaussian distribution.

**Usage**:
```python
noise = OrnsteinUhlenbeckProcess(theta=0.15, mu=0., sigma=0.3, dt=1e-2, size=1)
model.set(noise=noise)
```

---

These noise processes, such as `GaussianWhiteNoiseProcess` and `OrnsteinUhlenbeckProcess`, are typically used to introduce randomness during action selection in continuous action space reinforcement learning algorithms like DDPG. You can set them up as the noise generator in your RL agent's `set` function.These processes help in efficient exploration by generating noise that is added to the agent’s actions during training.

---

# Building a Custom Agent by Extending the RL Base Class:

This example demonstrates how to construct a reinforcement learning (RL) agent by extending a custom `RL` base class. The implementation uses both `Model` and `RL` classes to structure the agent modularly. Here, `Model` serves as a neural network wrapper, while `RL` manages RL-specific components.

**Step 1: Import `nn` and Define the Neural Network (Q-network) Class**

In this step, we start by importing `nn` from `Note`, a module that provides layer utilities and parameter management. The `Qnet` class, which inherits from the `Model` base class, uses `nn` layers for efficient Q-network construction.

```python
from Note import nn

class Qnet(nn.Model):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self, x):
        x = self.dense2(self.dense1(x))
        return x
```

Here, the `Model` superclass provides foundational methods for defining layers and managing parameters, making the setup of complex architectures more straightforward.

**Step 2: Create the DQN Agent by Extending the RL Class and Set Up the Environment**

The `DQN` class represents the agent, inheriting core reinforcement learning functionalities by extending the `RL` base class.

The agent’s Q-network and target network, `q_net` and `target_q_net`, are constructed using the `Qnet` class, where `nn` provides functions to define dense layers, activations, and to handle parameters more conveniently. Additionally, `self.env` initializes the "CartPole-v0" environment from `gym` directly within the agent, so it has a predefined environment for interaction.

```python
import gym

class DQN(nn.RL):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.param = self.q_net.param  # Parameters managed by `nn`
        self.env = gym.make('CartPole-v0')  # Environment created within the agent class
    
    def action(self, s):
        return self.q_net(s)
    
    def __call__(self, s, a, next_s, r, d):
        a = tf.expand_dims(a, axis=1)
        q_value = tf.gather(self.q_net(s), a, axis=1, batch_dims=1)
        next_q_value = tf.reduce_max(self.target_q_net(next_s), axis=1)
        target = tf.cast(r, 'float32') + 0.98 * next_q_value * (1 - tf.cast(d, 'float32'))
        TD = (q_value - target)
        return tf.reduce_mean(TD ** 2)
    
    def update_param(self):
        nn.assign_param(self.target_q_net.param, self.param)
```

**Explanation of Methods**

- **`action` Method**: This method takes the current state `s` as input and computes the Q-values using the Q-network (`q_net`). In the context of the RL class, the action method provides output for the RL class to select actions based on the policy. It returns the predicted Q-values for each possible action, which can then be used to determine the best action to take according to the agent's policy. This function effectively allows the agent to decide its next move based on learned values, facilitating exploration and exploitation.

- **`__call__` Method**: This method defines the loss calculation for DQN. It computes the Temporal Difference (TD) error by comparing the Q-value of the chosen action against the target Q-value. The target Q-value is derived from the reward and the maximum Q-value in the next state, adjusted by the discount factor.

- **`update_param` Method**: This method updates the parameters of the target Q-network (`target_q_net`) with those of the main Q-network (`q_net`). It ensures that the target network stays slightly behind the main network, stabilizing the training by providing more consistent target values.

Using `nn`, the `RL` base class handles much of the reinforcement learning logic, like parameter updates and replay buffer management, streamlining the creation of a DQN agent.

**Step 3: Initialize the Model and Train the Agent**

After defining both `Qnet` and `DQN`, we can instantiate the agent, set hyperparameters, and begin training using the `RL` class’s `train` method. The `train` method simplifies the training loop and efficiently manages data collection and updates.

```python
import tensorflow as tf
from Note.RL import rl

model = DQN(4, 128, 2)
model.set(policy=rl.EpsGreedyQPolicy(0.01), pool_size=10000, batch=64, update_steps=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=False)
``` 

This setup showcases how `nn`, `Model`, and `RL` components work together to streamline the development of reinforcement learning agents.

**HER(Hindsight Experience Replay):**

**Creating the `reward_done_func` function**:
   - `reward_done_func` is a custom reward function used to determine whether the agent has reached its goal and to provide an appropriate reward. In HER, this function also considers “substitute goals” (i.e., the states the agent actually reached) to dynamically adjust the reward. The function calculates reward values based on the agent’s distance from the goal (or other criteria) and determines whether the episode should end.

To enable a RL-based agent to support HER, an additional `reward_done_func` function needs to be defined.

**MARL(Multi-agent reinforcement learning):**

**Creating the `reward_done_func_ma` function**:
   - In multi-agent environments, each agent may have its own reward function and criteria for completion, depending on individual or team-based goals. The `reward_done_func_ma` can be adapted to multi-agent scenarios to compute rewards and evaluate termination conditions for each agent based on their interactions and objectives. This function ensures that agents receive rewards tailored to their specific goals, supporting individual learning.

To enable a RL-based agent to support MARL, an additional `reward_done_func_ma` function needs to be defined.

# Save model parameters:
```python
import pickle
output_file=open('param.dat','wb')
pickle.dump(model.param,output_file)
output_file.close()
```
or
```python
model = MyModel(...)
model.save_param('param.dat')
```

# Restore model parameters:
```python
import pickle
input_file=open('param.dat','rb')
param=pickle.load(input_file)
input_file.close()
```
or
```python
model = MyModel(...)
model.restore_param('param.dat')
```
or
```python
from Note import nn
param=nn.restore_param('param.dat')
```

# Save model:
```python
model = MyModel(...)
model.save('model.dat')
```

# Restore model:
```python
# distributed training
with strategy.scope():
    model = MyModel(...)
    model.restore('model.dat')
```
or
```python
model = MyModel(...)
model.restore('model.dat')
```

# LRFinder_rl:
**Usage:**

Create a Note agent, then execute this code:
```python
from Note import nn
# agent is a Note agent
agent.optimizer = tf.keras.optimizers.Adam()
lr_finder = nn.LRFinder_rl(agent)

# Train a agent with 77 episodes
# with learning rate growing exponentially from 0.0001 to 1
# N: Total number of iterations (or mini-batch steps) over which the learning rate is increased.
#    This parameter determines how many updates occur between the starting learning rate (start_lr)
#    and the ending learning rate (end_lr). The learning rate is increased exponentially by a fixed
#    multiplicative factor computed as:
#         factor = (end_lr / start_lr) ** (1.0 / N)
#    This ensures that after N updates, the learning rate will reach exactly end_lr.
#
# window_size: The size of the sliding window (i.e., the number of most recent episodes)
#              used to compute the moving average and standard deviation of the rewards.
#              This normalization helps smooth out the reward signal and adjust for the fact that
#              early episodes may have lower rewards (due to limited experience) compared to later ones.
#              By using only the recent window_size rewards, we obtain a more stable and current estimate
#              of the reward statistics for normalization.
lr_finder.find(train_loss, pool_network=False, N=77, window_size=7, start_lr=0.0001, end_lr=1, episodes=77)
```
or
```python
from Note import nn
# agent is a Note agent
agent.optimizer = tf.keras.optimizers.Adam()
strategy = tf.distribute.MirroredStrategy()
lr_finder = nn.LRFinder_rl(agent)

# Train a agent with 77 episodes
# with learning rate growing exponentially from 0.0001 to 1
# N: Total number of iterations (or mini-batch steps) over which the learning rate is increased.
#    This parameter determines how many updates occur between the starting learning rate (start_lr)
#    and the ending learning rate (end_lr). The learning rate is increased exponentially by a fixed
#    multiplicative factor computed as:
#         factor = (end_lr / start_lr) ** (1.0 / N)
#    This ensures that after N updates, the learning rate will reach exactly end_lr.
#
# window_size: The size of the sliding window (i.e., the number of most recent episodes)
#              used to compute the moving average and standard deviation of the rewards.
#              This normalization helps smooth out the reward signal and adjust for the fact that
#              early episodes may have lower rewards (due to limited experience) compared to later ones.
#              By using only the recent window_size rewards, we obtain a more stable and current estimate
#              of the reward statistics for normalization.
lr_finder.find(pool_network=False, strategy=strategy, N=77, window_size=7, start_lr=0.0001, end_lr=1, episodes=77)
```
```python
# Plot the reward, ignore 20 batches in the beginning and 5 in the end
lr_finder.plot_reward(n_skip_beginning=20, n_skip_end=5)
```
```python
# Plot rate of change of the reward
# Ignore 20 batches in the beginning and 5 in the end
# Smooth the curve using simple moving average of 20 batches
# Limit the range for y axis to (-0.02, 0.01)
lr_finder.plot_reward_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
```

# OptFinder_rl:
**Usage:**

Create a Note agent, then execute this code:
```python
from Note import nn
# agent is a Note agent
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
opt_finder = nn.OptFinder_rl(agent, optimizers)

# Train a agent with 7 episodes
opt_finder.find(train_loss, pool_network=False, episodes=7)
```
or
```python
from Note import nn
# agent is a Note agent
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
strategy = tf.distribute.MirroredStrategy()
opt_finder = nn.OptFinder_rl(agent, optimizers)

# Train a agent with 7 episodes
opt_finder.find(pool_network=False, strategy=strategy, episodes=7)
```

# ParallelFinder:

**Overview**

The **AgentFinder** class is designed for reinforcement learning or multi-agent training scenarios. It trains multiple agents in parallel and selects the best performing agent based on a chosen metric (reward or loss). The class employs multiprocessing to run each agent’s training in its own process and uses callbacks at the end of each episode to update performance logs. Depending on the selected metric, at the end of the training episodes, it computes the mean reward or mean loss for each agent and updates the shared logs with the best optimizer and corresponding performance value.

---

**Key Attributes**

- **agents**  
  *Type:* `list`  
  *Description:* A list of agent instances to be trained. Each agent will run its training in a separate process.

- **optimizers**  
  *Type:* `list`  
  *Description:* A list of optimizers corresponding to the agents, used during the training process.

- **rewards**  
  *Type:* Shared dictionary (created via `multiprocessing.Manager().dict()`)  
  *Description:* Records the reward values for each episode for every agent. For each agent, a list of rewards is maintained.

- **losses**  
  *Type:* Shared dictionary  
  *Description:* Records the loss values for each episode for every agent. For each agent, a list of losses is maintained.

- **logs**  
  *Type:* Shared dictionary  
  *Description:* Stores key training information. Initially, it contains:
  - `best_reward`: Set to a very low value (-1e9) to store the best mean reward.
  - `best_loss`: Set to a high value (1e9) to store the lowest mean loss.
  - When training is complete, it also stores `best_opt`, which corresponds to the optimizer of the best performing agent.

- **lock**  
  *Type:* `multiprocessing.Lock`  
  *Description:* A multiprocessing lock used to ensure data consistency and thread safety when multiple processes update the shared dictionaries.

- **episode**  
  *Type:* `int`  
  *Description:* The total number of training episodes, set in the `find` method. This value is used to determine if the current episode is the final one.

---

**Main Methods**

**1. `__init__(self, agents, optimizers)`**

**Purpose:**  
Initializes an AgentFinder instance by setting the list of agents and corresponding optimizers. It also creates shared dictionaries for rewards, losses, and logs, and initializes a multiprocessing lock to ensure safe data access.

**Parameters:**
- `agents`: A list of agent instances.
- `optimizers`: A list of optimizers corresponding to the agents.

**Details:**  
The constructor uses `multiprocessing.Manager()` to create shared dictionaries (`rewards`, `losses`, `logs`) and sets initial values for best reward and best loss for subsequent comparisons. A lock object is created to synchronize updates in a multiprocessing environment.

**2. `on_episode_end(self, episode, logs, agent=None, lock=None)`**

**Purpose:**  
This callback function is invoked at the end of each episode when the metric is set to 'reward'. It updates the corresponding agent’s reward list and, if the episode is the last one, calculates the mean reward. If the mean reward exceeds the current best reward recorded in the shared logs, it updates the logs with the new best reward and the corresponding optimizer.

**Parameters:**
- `episode`: The current episode number (starting from 0).
- `logs`: A dictionary containing training information for the current episode; it must include the key `'reward'`.
- `agent`: The current agent instance, used to update the reward list and access its optimizer.
- `lock`: The multiprocessing lock used to synchronize access to shared data.

**Key Logic:**
1. Acquire the lock with `lock.acquire()` to ensure safe data updates.
2. Retrieve the current episode’s reward from `logs`.
3. Append the reward to the corresponding agent’s list in the `rewards` dictionary.
4. If this is the last episode (i.e., `episode + 1 == self.episode`), calculate the mean reward.
5. If the mean reward is higher than the current `best_reward` in the shared logs, update `logs['best_reward']` and `logs['best_opt']` (using the agent’s optimizer).
6. Release the lock using `lock.release()`.

**3. `on_episode_end_(self, episode, logs, agent=None, lock=None)`**

**Purpose:**  
This callback function is used when the metric is set to 'loss'. It updates the corresponding agent’s loss list and, at the end of the final episode, computes the mean loss. If the mean loss is lower than the current best loss recorded in the shared logs, it updates the logs with the new best loss and the corresponding optimizer.

**Parameters:**
- `episode`: The current episode number (starting from 0).
- `logs`: A dictionary containing training information for the current episode; it must include the key `'loss'`.
- `agent`: The current agent instance.
- `lock`: The multiprocessing lock used to synchronize access to shared data.

**Key Logic:**
1. Acquire the lock to ensure safe updates.
2. Retrieve the loss from `logs` and append it to the corresponding agent’s list in the `losses` dictionary.
3. At the last episode, calculate the mean loss and compare it to the current best loss.
4. If the mean loss is lower, update `logs['best_loss']` and `logs['best_opt']` (with the agent’s optimizer).
5. Release the lock.

**4. `find(self, train_loss=None, pool_network=True, processes=None, processes_her=None, processes_pr=None, strategy=None, episodes=1, metrics='reward', jit_compile=True)`**

**Purpose:**  
Starts the training of multiple agents using multiprocessing and utilizes callback functions to update the best agent information based on the selected metric (reward or loss).

**Parameters:**
- `train_loss`: A function or parameter for computing the training loss (optional).
- `pool_network`: Boolean flag indicating whether to use a shared network pool.
- `processes`: Number of processes to be used for training (optional).
- `processes_her`: Parameters related to HER (Hindsight Experience Replay) (optional).
- `processes_pr`: Parameters possibly related to Prioritized Experience Replay (optional).
- `strategy`: Distributed training strategy (optional). If provided, the distributed training mode is used; otherwise, standard training is performed.
- `episodes`: Total number of training episodes.
- `metrics`: The metric to be used, either `'reward'` or `'loss'`. This choice determines which callback function is used.
- `jit_compile`: Boolean flag indicating whether to enable JIT compilation to speed up training.

**Key Logic:**
1. Set the total number of episodes to `self.episodes`.
2. Iterate over each agent:
   - If the selected metric is `'reward'`:
     - Use `functools.partial` to create a `partial_callback` that binds the agent, lock, and the `on_episode_end` callback.
     - Create a callback instance using `nn.LambdaCallback`.
     - Initialize the agent’s reward list in the `rewards` dictionary.
   - If the selected metric is `'loss'`:
     - Similarly, bind the `on_episode_end_` callback.
     - Initialize the agent’s loss list in the `losses` dictionary.
3. Assign the corresponding optimizer to each agent.
4. Depending on whether a `strategy` is provided, choose the training mode:
   - If `strategy` is `None`, call the agent’s `train` method with the appropriate parameters (e.g., training loss, episodes, network pool options, process parameters, callbacks, and jit_compile settings).
   - If a `strategy` is provided, call the agent’s `distributed_training` method with similar parameters and a similar callback setup.
5. Start all training processes and wait for them to complete using `join()`.

---

**Example Usage**

Below is an example demonstrating how to use AgentFinder to train multiple agents and select the best performing agent based on either reward or loss:

```python
from Note import nn

# Assume agent1 and agent2 are two initialized agent instances,
# and optimizer1 and optimizer2 are their respective optimizers.
agent1 = ...  # Initialize agent 1
agent2 = ...  # Initialize agent 2
optimizer1 = ...  # Optimizer for agent 1
optimizer2 = ...  # Optimizer for agent 2

# Create lists of agents and optimizers
agents = [agent1, agent2]
optimizers = [optimizer1, optimizer2]

# Initialize the AgentFinder instance
parallel_finder = nn.ParallelFinder_rl(agents, optimizers)

# Assume train_loss is defined as a function or metric for calculating training loss (if needed)
train_loss = ...

# Choose the evaluation metric: 'reward' or 'loss'
metrics_choice = 'reward'  # or 'loss'

# Execute training with 10 episodes and enable JIT compilation
parallel_finder.find(
    train_loss=train_loss,
    pool_network=True,
    processes=4,
    processes_her=2,
    processes_pr=2,
    strategy=None,  # Pass None to use standard training (not distributed)
    episodes=10,
    metrics=metrics_choice,
    jit_compile=True
)

# After training, retrieve the best record from agent_finder.logs
if metrics_choice == 'reward':
    print("Best Mean Reward:", agent_finder.logs['best_reward'])
else:
    print("Best Mean Loss:", agent_finder.logs['best_loss'])
print("Best Optimizer:", agent_finder.logs['best_opt'])
```
