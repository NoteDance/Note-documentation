# Train:
## Note and Keras:
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
model.train(train_loss, optimizer, 100)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
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
# Use PPO.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.PPO import PPO
# from Note.models.docs_example.RL.keras.PPO import PPO

model=PPO(4,128,2,0.7,0.7)
model.set(policy=rl.SoftmaxPolicy(),pool_size=10000,batch=64,update_steps=1000,PPO=True)
optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
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
model.train(train_loss, optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning.
import tensorflow as tf
from Note.RL import rl
from Note.models.docs_example.RL.note.MADDPG import DDPG
# from Note.models.docs_example.RL.keras.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
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
## PyTorch:
Agent built with PyTorch.
```python
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DQN import DQN

model=DQN(4,128,2)
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(optimizer, 100)

# If use prioritized replay.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
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
# Use HER.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning.
import torch
from Note.RL import rl
from Note.models.docs_example.RL.pytorch.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 100)
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
## MirroredStrategy:
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
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
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

model.set(policy=rl.SoftmaxPolicy(),pool_size=10000,update_steps=1000,PPO=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)
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

model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,criterion=-5,trial_count=10,HER=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 2000)
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

model.set(policy=rl.SoftmaxPolicy(),pool_size=3000,trial_count=10,MA=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100)
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
model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100, pool_network=True, processes=7)
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
model.set(noise=rl.GaussianWhiteNoiseProcess(),pool_size=10000,trial_count=10,HER=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 2000, pool_network=True, processes=7, processes_her=4)
```
## MultiWorkerMirroredStrategy:
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

multi_worker_model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
multi_worker_model.distributed_training(global_batch_size, optimizer, strategy, num_episodes=100,
                    pool_network=True, processes=7)

# If set criterion.
# model.set(policy=rl.EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
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

# `RL.set`:

## **Function Description**:
The `set` function configures various parameters of the Reinforcement Learning (RL) agent. These parameters control the policy, noise, experience pool, batch size, update frequency, and training termination conditions. By adjusting these settings, users can fine-tune the agent's behavior and training process to suit specific RL tasks.

---

## **Function Signature**:
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
        MA=False, 
        PR=False, 
        epsilon=None, 
        initial_TD=7., 
        alpha=0.7):
```

---

## **Parameter Description**:

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

- **`MA`** (`bool`):  
  Whether to use Multi-Agent (MA) reinforcement learning.

- **`PR`** (`bool`):  
  Whether to use Prioritized Experience Replay (PR), a technique to sample experiences based on their significance.

- **`epsilon`** (`float` or `None`):  
  The `ε` value used in an `ε-greedy` policy, controlling the probability of choosing a random action to encourage exploration.

- **`initial_TD`** (`float`):  
  The initial TD-error value used in Prioritized Replay. A higher TD-error leads to higher prioritization of the sample in the experience pool.

- **`alpha`** (`float`):  
  The `α` value used in Prioritized Replay, determining how much the TD-error influences sample prioritization. A higher `α` increases the importance of prioritizing higher TD-error experiences.

---

## **Usage Example**:

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

# `Policy classes`:

## **SoftmaxPolicy**

### **Description**:
Implements a softmax policy for multinomial distribution. This policy selects actions based on a probability distribution, where each action has a probability proportional to the exponentiated Q-value or reward estimate.

### **Methods**:

- **`select_action(nb_actions, probs)`**:
  - **Arguments**:
    - `nb_actions` (`int`): Number of possible actions.
    - `probs` (`np.ndarray`): A probability distribution over the actions.
  - **Returns**:
    - `action` (`int`): The action selected according to the probability distribution `probs`.

### **Usage**:
Use `SoftmaxPolicy` with the `set` function of an RL agent:
```python
policy = SoftmaxPolicy()
model.set(policy=policy)
```

---

## **EpsGreedyQPolicy**

### **Description**:
Implements the epsilon-greedy policy. With probability `epsilon`, this policy selects a random action to encourage exploration. Otherwise, it selects the action with the highest estimated Q-value.

### **Constructor**:

- **`__init__(eps=.1)`**:
  - **Arguments**:
    - `eps` (`float`): Epsilon value representing the probability of choosing a random action. Defaults to `0.1`.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action, either random (with probability `eps`) or the best action (with probability `1 - eps`).

### **Usage**:
Use `EpsGreedyQPolicy` with the `set` function of an RL agent:
```python
policy = EpsGreedyQPolicy(eps=0.05)
model.set(policy=policy)
```

---

## **GreedyQPolicy**

### **Description**:
Implements the greedy policy, where the agent always selects the action with the highest estimated Q-value. This policy does not explore other actions.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The action with the highest Q-value.

### **Usage**:
Use `GreedyQPolicy` with the `set` function of an RL agent:
```python
policy = GreedyQPolicy()
model.set(policy=policy)
```

---

## **BoltzmannQPolicy**

### **Description**:
Implements the Boltzmann Q Policy. This policy selects actions based on a probability distribution derived from exponentiated Q-values, where higher Q-values have higher probabilities. The `tau` parameter controls the exploration: higher `tau` values result in more exploration, while lower values focus on exploitation.

### **Constructor**:

- **`__init__(tau=1., clip=(-500., 500.))`**:
  - **Arguments**:
    - `tau` (`float`): Temperature parameter controlling exploration. Default is `1.0`.
    - `clip` (`tuple`): Range to clip the Q-values before exponentiation. Default is `(-500., 500.)`.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action according to the Boltzmann distribution of Q-values.

### **Usage**:
Use `BoltzmannQPolicy` with the `set` function of an RL agent:
```python
policy = BoltzmannQPolicy(tau=0.5)
model.set(policy=policy)
```

---

## **MaxBoltzmannQPolicy**

### **Description**:
Combines epsilon-greedy and Boltzmann Q-policy. With probability `epsilon`, the agent follows the Boltzmann distribution to select an action. With probability `1 - epsilon`, it selects the action with the highest Q-value.

### **Constructor**:

- **`__init__(eps=.1, tau=1., clip=(-500., 500.))`**:
  - **Arguments**:
    - `eps` (`float`): Epsilon value for selecting random actions.
    - `tau` (`float`): Temperature parameter for Boltzmann exploration.
    - `clip` (`tuple`): Range to clip the Q-values. Default is `(-500., 500.)`.

### **Methods**:

- **`select_action(q_values)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
  - **Returns**:
    - `action` (`int`): The selected action, either based on Boltzmann exploration or the greedy choice.

### **Usage**:
Use `MaxBoltzmannQPolicy` with the `set` function of an RL agent:
```python
policy = MaxBoltzmannQPolicy(eps=0.1, tau=0.5)
model.set(policy=policy)
```

---

## **BoltzmannGumbelQPolicy**

### **Description**:
Implements the Boltzmann-Gumbel exploration policy, which is invariant to the mean of rewards but sensitive to reward variance. This policy uses Gumbel noise to perturb the Q-values for exploration and adapts over time based on the parameter `C`.

### **Constructor**:

- **`__init__(C=1.0)`**:
  - **Arguments**:
    - `C` (`float`): Exploration parameter to adjust for variance in rewards.

### **Methods**:

- **`select_action(q_values, step_counter)`**:
  - **Arguments**:
    - `q_values` (`np.ndarray`): Q-value estimates for each action.
    - `step_counter` (`int`): Current step of the training process.
  - **Returns**:
    - `action` (`int`): The selected action based on Boltzmann-Gumbel exploration.

### **Usage**:
Use `BoltzmannGumbelQPolicy` with the `set` function of an RL agent:
```python
policy = BoltzmannGumbelQPolicy(C=1.0)
model.set(policy=policy)
```

---

## **GumbelSoftmaxPolicy**

### **Description**:
Implements the Gumbel Softmax policy for continuous action spaces. This policy samples from a Gumbel distribution and returns one-hot encoded actions for discrete action selection.

### **Constructor**:

- **`__init__(temperature=1.0, eps=0.01)`**:
  - **Arguments**:
    - `temperature` (`float`): Temperature parameter for Gumbel sampling.
    - `eps` (`float`): Epsilon value for exploration in the one-hot encoding process.

### **Methods**:

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

### **Usage**:
Use `GumbelSoftmaxPolicy` with the `set` function of an RL agent:
```python
policy = GumbelSoftmaxPolicy(temperature=0.5, eps=0.01)
model.set(policy=policy)
```

---

This documentation provides detailed descriptions and usage examples for each policy class and reflects the typical way of passing these policies to an RL agent using the `set` function.

# `Noise classes`:

## **GaussianWhiteNoiseProcess**

### **Description**:
Implements a Gaussian white noise process, generating noise from a Gaussian distribution with mean `mu` and time-varying standard deviation `sigma` (which anneals over time). This type of noise is commonly used in exploration strategies for continuous action spaces.

### **Constructor**:
- **`__init__(mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1)`**:
  - **Arguments**:
    - `mu` (`float`): Mean of the Gaussian distribution.
    - `sigma` (`float`): Initial standard deviation.
    - `sigma_min` (`float`): Minimum standard deviation after annealing. If `None`, annealing is disabled.
    - `n_steps_annealing` (`int`): Number of steps over which the annealing occurs.
    - `size` (`int`): Size of the noise vector to be sampled.

### **Methods**:

- **`sample()`**:
  - Generates a sample of Gaussian noise based on the current standard deviation (`sigma`), which anneals over time.
  - **Returns**:
    - `sample` (`np.ndarray`): Sampled noise from the Gaussian distribution.

### **Usage**:
```python
noise = GaussianWhiteNoiseProcess(mu=0., sigma=1., sigma_min=0.1, n_steps_annealing=1000, size=1)
model.set(noise=noise)
```

---

## **OrnsteinUhlenbeckProcess**

### **Description**:
This process generates noise using the Ornstein-Uhlenbeck process, a continuous-time stochastic process often used to model time-correlated noise. It is frequently applied in reinforcement learning, especially for exploration in environments with continuous action spaces (e.g., DDPG). The noise tends to revert to the mean over time, controlled by the parameter `theta`.

### **Constructor**:
- **`__init__(theta, mu=0., sigma=1., dt=1e-2, size=1, sigma_min=None, n_steps_annealing=1000)`**:
  - **Arguments**:
    - `theta` (`float`): Rate of mean reversion (higher `theta` means stronger pull towards `mu`).
    - `mu` (`float`): Mean value to which the process reverts.
    - `sigma` (`float`): Initial standard deviation of the noise.
    - `dt` (`float`): Time step for discretization.
    - `size` (`int`): Size of the noise vector.
    - `sigma_min` (`float`): Minimum standard deviation for annealing.
    - `n_steps_annealing` (`int`): Number of steps over which the standard deviation anneals.

### **Methods**:

- **`sample()`**:
  - Generates a noise sample based on the current state of the process and updates the internal state.
  - **Returns**:
    - `x` (`np.ndarray`): Sampled noise from the Ornstein-Uhlenbeck process.

- **`reset_states()`**:
  - Resets the internal state (`x_prev`) of the process to a random value drawn from a Gaussian distribution.

### **Usage**:
```python
noise = OrnsteinUhlenbeckProcess(theta=0.15, mu=0., sigma=0.3, dt=1e-2, size=1)
model.set(noise=noise)
```

---

These noise processes, such as `GaussianWhiteNoiseProcess` and `OrnsteinUhlenbeckProcess`, are typically used to introduce randomness during action selection in continuous action space reinforcement learning algorithms like DDPG. You can set them up as the noise generator in your RL agent's `set` function.These processes help in efficient exploration by generating noise that is added to the agent’s actions during training.
