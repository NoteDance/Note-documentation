# Introduction:
A comprehensive TensorFlow-based model training and management framework with support for distributed training and flexible model architecture.


# Table of Contents

- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Training](#training)
  - [Basic Training](#basic-training)
  - [Distributed Training](#distributed-training)
  - [Adaptive Batch Sizing](#adaptive-batch-sizing)
- [Model Management](#model-management)
  - [Saving & Loading](#saving--loading)
  - [Fine-tuning](#fine-tuning)
- [Advanced Features](#advanced-features)
  - [Namespace Management](#namespace-management)
  - [Layer Freezing](#layer-freezing)
  - [Custom Callbacks](#custom-callbacks)
- [API Reference](#api-reference)

---

# Quick Start

```python
from Note import nn
import tensorflow as tf

# Define your model
class MyModel(nn.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.conv2d(32, 3, activation='relu')
        self.flatten = nn.flatten()
        self.d1 = nn.dense(128, activation='relu')
        self.d2 = nn.dense(10)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Prepare datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Initialize model and training components
model = MyModel()
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Train
model.train(train_ds, loss_object, train_loss, optimizer, epochs=5,
            train_accuracy=train_accuracy, test_ds=test_ds,
            test_loss=test_loss, test_accuracy=test_accuracy)
```

---

# Model Architecture

## Building Custom Models

Inherit from `nn.Model` and define your architecture:

```python
from Note import nn

class CustomModel(nn.Model):
    def __init__(self):
        super().__init__()
        # Define layers
        self.conv1 = nn.conv2d(64, 3, activation='relu')
        self.pool = nn.max_pool2d(2, 2)
        self.flatten = nn.flatten()
        self.dense1 = nn.dense(256, activation='relu')
        self.dropout = nn.dropout(0.5)
        self.output = nn.dense(10)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output(x)
```

## Model Summary

```python
model = CustomModel()
model.summary()  # Display parameter count and memory usage
```

**Output:**
```
Model Summary
-------------
Total params: 407050 (1.55 MB)
Trainable params: 407050 (1.55 MB)
Non-trainable params: 0 (0.00 Bytes)
```

---

# Training

## Basic Training

```python
model.train(
    train_ds=train_ds,
    loss_object=loss_object,
    train_loss=train_loss,
    optimizer=optimizer,
    epochs=10,
    train_accuracy=train_accuracy,
    test_ds=test_ds,
    test_loss=test_loss,
    test_accuracy=test_accuracy,
    jit_compile=True  # Enable JIT compilation for faster training
)
```

## Training with Early Stopping

```python
model.end_acc = 0.95  # Stop when accuracy reaches 95%
# or
model.end_loss = 0.1  # Stop when loss drops below 0.1

model.train(train_ds, loss_object, train_loss, optimizer, epochs=100,
            train_accuracy=train_accuracy)
```

## Model Checkpointing

```python
# Save every epoch
model.path = 'model.dat'
model.save_freq = 1
model.max_save_files = 3  # Keep only last 3 checkpoints

# Save every N batches
model.save_freq_ = 1875  # Save every 1875 batches

# Save only best model
model.save_top_k = 2
model.monitor = 'val_loss'  # or 'val_accuracy'

# Save parameters only (smaller file size)
model.save_param_only = True

model.train(train_ds, loss_object, train_loss, optimizer, epochs=10)
```

## Visualization

```python
# Visualize training metrics
model.visualize_train()      # Training loss and accuracy
model.visualize_test()        # Test loss and accuracy
model.visualize_comparison()  # Compare training vs test metrics
```

---

# Distributed Training

## MirroredStrategy (Multi-GPU on Single Machine)

```python
strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)
).shuffle(10000).batch(GLOBAL_BATCH_SIZE)

with strategy.scope():
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

model.distributed_training(
    train_dataset=train_dataset,
    loss_object=loss_object,
    global_batch_size=GLOBAL_BATCH_SIZE,
    optimizer=optimizer,
    strategy=strategy,
    epochs=10,
    train_accuracy=train_accuracy
)
```

## MultiWorkerMirroredStrategy (Multi-Machine)

```python
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = MyModel()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.distributed_training(
    train_dataset=train_dataset,
    loss_object=loss_object,
    global_batch_size=global_batch_size,
    optimizer=optimizer,
    strategy=strategy,
    num_epochs=10,
    num_steps_per_epoch=70
)
```

## ParameterServerStrategy

```python
strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)

coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)

with strategy.scope():
    model = MyModel()
    optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)

model.distributed_training(
    loss_object=loss_object,
    optimizer=optimizer,
    strategy=strategy,
    num_epochs=7,
    num_steps_per_epoch=7,
    dataset_fn=dataset_fn,
    test_dataset_fn=test_dataset_fn
)
```

---

# Model Management

## Saving & Loading

**Save/Load Complete Model:**
```python
# Save
model.save('model.dat')

# Load
model = MyModel()
model.restore('model.dat')
```

**Save/Load Parameters Only:**
```python
# Save
model.save_param('params.dat')

# Load
model = MyModel()
model.restore_param('params.dat')

# Or use pickle directly
import pickle
with open('params.dat', 'rb') as f:
    params = pickle.load(f)
nn.assign_param(model.param, params)
```

**Get Model Configuration:**
```python
info = model.get_info()
# Returns dict with training configuration, hyperparameters, etc.
```

## Fine-tuning

```python
# Load pre-trained model
model = PretrainedModel()
model.restore('pretrained.dat')

# Replace output layer for new task (10 classes)
model.fine_tuning(num_classes=10, flag=0)

# Freeze pre-trained layers
optimizer.learning_rate = 0.0001

# Fine-tune on new dataset
model.train(fine_tune_ds, loss_object, train_loss, optimizer, epochs=5)

# Optionally unfreeze all layers
model.fine_tuning(num_classes=10, flag=1)
```

**Fine-tuning Flags:**
- `flag=0`: Replace head, freeze base layers
- `flag=1`: Unfreeze all layers
- `flag=2`: Restore original head

---

# Advanced Features

## Namespace Management

Organize layers into namespaces for fine-grained control:

```python
class Block:
    def __init__(self, name):
        nn.Model.add()
        nn.Model.namespace(name)
        self.layer1 = nn.dense(128, activation='relu')
        self.layer2 = nn.dense(128, activation='relu')
        nn.Model.namespace()  # Close namespace
        nn.Model.apply(self.init_weights)
    
    def init_weights(self, layer):
        if isinstance(layer, nn.dense):
            layer.weight.assign(nn.trunc_normal_(layer.weight, std=0.02))
    
    def __call__(self, x):
        return self.layer2(self.layer1(x))

class Model(nn.Model):
    def __init__(self):
        super().__init__()
        self.block1 = Block('block1')
        self.block2 = Block('block2')
```

## Layer Freezing

```python
# Freeze specific namespace
model.freeze('block1')

# Unfreeze specific namespace
model.unfreeze('block1')

# Freeze all layers
model.freeze()

# Unfreeze all layers
model.unfreeze()
```

## Training/Evaluation Mode

```python
# Set entire model to evaluation mode
model.training(flag=False)

# Set entire model to training mode
model.training(flag=True)

# Set specific namespace to eval mode
model.eval('block1', flag=True)

# Set specific namespace to training mode
model.eval('block1', flag=False)
```

## Adaptive Batch Sizing

Automatically adjust batch size based on gradient variance:

```python
# Define adaptive batch sizing function
def batch_size_fn(train_ds):
    return model.adabatch(
        train_ds=train_ds,
        num_samples=10,
        target_noise=1e-3,
        scale=1.0,
        min_batch=16,
        max_batch=256,
        lr_params={'lr_rate': 0.1, 'min': 1e-5, 'max': 1e-2},
        buffer_size=10000
    )

# Register function
model.batch_size_fn = batch_size_fn

# Train with adaptive batching
model.train(train_ds, loss_object, train_loss, optimizer, epochs=10)
```

## Parameter Type Casting

```python
# Cast all parameters to float16
model.cast_param(dtype=tf.float16)

# Cast specific parameter group
model.cast_param(key='dense_weight', dtype=tf.float32)
```

## Weight Decay

```python
# Apply weight decay to dense layers
model.apply_decay('dense_weight', weight_decay=0.9, flag=True)

# Remove weight decay
model.apply_decay('dense_weight', weight_decay=0.9, flag=False)
```

---

# API Reference

## Model Methods

| Method                          | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `train()`                       | Single-device training loop with **Prioritized Experience Replay** (PER), **parallel training & validation**, and **parallel training & saving** support |
| `distributed_training()`        | Distributed training (`MirroredStrategy`, `MultiWorkerMirroredStrategy`, `ParameterServerStrategy`) with PER, parallel validation, and parallel saving support |
| `test()`                        | Evaluate on test dataset (supports multiprocessing-based parallel evaluation) |
| `save()` / `restore()`          | Save/load full model (architecture + parameters + optimizer state)           |
| `save_param()` / `restore_param()` | Save/load parameters only                                                |
| `summary()`                     | Print parameter counts, trainable/non-trainable stats, and memory usage      |
| `training(flag)`                | Set global training (`True`) or evaluation (`False`) mode                   |
| `freeze(name)` / `unfreeze(name)` | Freeze/unfreeze parameters by namespace (or all if `name=None`)            |
| `eval(name, flag)`              | Set namespace (or all) to evaluation (`flag=True`) or training mode         |
| `fine_tuning(num_classes, flag)`| Replace head and control backbone freezing for transfer learning           |
| `cast_param(key, dtype)`        | Cast parameter data types (all or selectively by key)                       |
| `visualize_train()`             | Plot training loss/accuracy curves                                           |
| `visualize_test()`              | Plot validation/test loss/accuracy curves                                   |
| `visualize_comparison()`        | Overlay train vs validation/test curves                                      |
| `adabatch()`                    | Adaptive batch size adjustment based on gradient noise                      |
| `get_info()`                    | Return dictionary of current training/configuration state                   |

## Training Parameters (`train()` and `distributed_training()`)

Both methods share the same core parameters. `distributed_training()` adds strategy-specific arguments (`strategy`, `global_batch_size`, `num_steps_per_epoch`, etc.).

| Parameter                    | Type                     | Default | Description                                                                 |
|------------------------------|--------------------------|---------|-----------------------------------------------------------------------------|
| `train_ds` / `train_dataset` | `tf.data.Dataset`        | -       | Training dataset                                                            |
| `loss_object`                | `tf.keras.losses.Loss`   | -       | Loss function                                                               |
| `train_loss`                 | `tf.keras.metrics.Metric`| -       | Metric to track training loss                                               |
| `optimizer`                  | `tf.keras.optimizers.Optimizer` | `None` | Optimizer (can be set later)                                          |
| `epochs` / `num_epochs`      | `int` / `None`           | `None`  | Number of epochs (`None` → train indefinitely)                              |
| `train_accuracy`             | `tf.keras.metrics.Metric`| `None`  | Optional training accuracy metric                                           |
| `test_ds` / `test_dataset`   | `tf.data.Dataset`        | `None`  | Validation/test dataset (used when `parallel_training_and_test=False`)      |
| `test_loss`                  | `tf.keras.metrics.Metric`| `None`  | Validation loss metric                                                      |
| `test_accuracy`              | `tf.keras.metrics.Metric`| `None`  | Validation accuracy metric                                                  |
| `parallel_training_and_test` | `bool`                   | `False` | Run validation in separate process (non-blocking)                           |
| `parallel_training_and_save` | `bool`                   | `False` | Run checkpoint saving in separate process (non-blocking)                    |
| `parallel_dump`                 | `bool`                | `False`   | When `True` and combined with `parallel_training_and_save`, saves parameters and optimizer states **in parallel** to a **folder** (one file per variable) instead of a single `.dat` file. Ideal for extremely large models. |
| `test_data`                  | `np.ndarray` / `None`    | `None`  | Full validation data array (required when `parallel_training_and_test=True`)|
| `test_labels`                | `np.ndarray` / `None`    | `None`  | Full validation labels array (required when `parallel_training_and_test=True`)|
| `test_batch_size`            | `int` / `None`           | `None`  | Validation batch size (defaults to training batch size if `None`)           |
| `test_freq`                  | `int`                    | `1`     | Run validation every N epochs                                               |
| `PR`                         | `bool`                   | `False` | Enable **Prioritized Experience Replay**                                    |
| `train_data`                 | `np.ndarray` / `None`    | `None`  | Full training data array (required when `PR=True`)                          |
| `train_labels`               | `np.ndarray` / `None`    | `None`  | Full training labels array (required when `PR=True`)                         |
| `alpha`                      | `float` / `None`         | `None`  | Prioritization exponent α (used when `PR=True`)                             |
| `ess_threshold`              | `float` / `None`         | `None`  | Target effective sample size for dynamic update scaling                     |
| `scale`                      | `float`                  | `1.0`   | Scaling factor for update frequency adjustment                              |
| `num_updates`                | `int` / `None`           | `None`  | Base number of updates per PER cycle                                        |
| `min_num_updates`            | `int` / `None`           | `None`  | Minimum updates per cycle                                                   |
| `max_num_updates`            | `int` / `None`           | `None`  | Maximum updates per cycle                                                   |
| `processes`                  | `int` / `None`           | `None`  | Number of processes for parallel validation                                 |
| `jit_compile`                | `bool`                   | `True`  | Enable XLA/JIT compilation for train/test steps                             |
| `p`                          | `int` / `None`           | `None`  | Print frequency control (~every 10% of epochs by default)                   |

## Prioritized Experience Replay (PER)

When `PR=True`, training alternates between standard and prioritized sampling:

- **Even epochs**: Standard training, priorities updated based on loss
- **Odd epochs**: Prioritized sampling + adaptive number of updates based on Effective Sample Size (ESS)

**Example:**
```python
model.train(
    train_ds=train_ds,
    loss_object=loss_obj,
    train_loss=train_loss,
    optimizer=optimizer,
    epochs=20,
    PR=True,
    train_data=x_train,
    train_labels=y_train,
    alpha=0.6,
    ess_threshold=1000.0,
    scale=1.0,
    num_updates=200,
    min_num_updates=100,
    max_num_updates=400
)
```

## Parallel Training & Validation

When `parallel_training_and_test=True`, validation runs in a background process, allowing training to continue without blocking.

**Key Benefits:**
- Non-blocking validation → higher epoch throughput
- Ideal for large validation sets or expensive evaluation

**Required Parameters:**
- `test_data`, `test_labels`: Full validation arrays (NumPy)
- `test_batch_size`: Validation batch size
- `test_freq`: Evaluate every N epochs

**Example:**
```python
model.train(
    train_ds=train_ds,
    loss_object=loss_obj,
    train_loss=train_loss,
    optimizer=optimizer,
    epochs=50,
    parallel_training_and_test=True,
    test_data=x_val,
    test_labels=y_val,
    test_batch_size=128,
    test_freq=2  # Validate every 2 epochs
)
```

Validation metrics are collected asynchronously and logged when available.

## Parallel Training & Saving

When `parallel_training_and_save=True`, checkpoint saving runs in a background process, preventing I/O from blocking training.

**Key Benefits:**
- Non-blocking saves → no training pauses during checkpointing
- Useful for frequent saves or large models

**Behavior:**
- Saves are triggered normally (by `save_freq`, `save_freq_`, or best metric)
- Full model or parameters are copied and saved in a separate process
- File naming follows the same rules as regular saving

**Example:**
```python
model.train(
    train_ds=train_ds,
    loss_object=loss_obj,
    train_loss=train_loss,
    optimizer=optimizer,
    epochs=100,
    path='checkpoints/model.dat',
    save_freq=5,
    parallel_training_and_save=True
)
```

## Note on Parallel Dumping (`parallel_dump=True`)
- Creates a folder at the specified `path`.
- Each parameter and optimizer state is saved in a separate file (`param_X.dat`, `state_X.dat`).
- Significantly reduces memory pressure and enables saving of models too large for single-file pickling.
- Requires `parallel_training_and_save=True`.

## Model Attributes (Configuration)

Set directly on the model instance.

| Attribute                        | Type      | Default         | Description |
|----------------------------------|-----------|-----------------|-------------|
| `path`                           | `str`     | `None`          | Base path for saving checkpoints (file or directory) |
| `save_freq`                      | `int`     | `1`             | Save checkpoint every N epochs |
| `save_freq_`                     | `int`     | `None`          | Save checkpoint every N batches (overrides `save_freq`) |
| `max_save_files`                 | `int`     | `1`             | Maximum number of checkpoints to keep (when `save_top_k=None`) |
| `save_top_k`                     | `int`     | `1`             | Keep only the top K best checkpoints based on monitored metric (recommended) |
| `save_last`                      | `bool`    | `True`          | Additionally save the final epoch as `-last.dat` (or `-last` folder) |
| `save_best_only`                 | `bool`    | `False`         | Save only when the monitored metric improves |
| `monitor`                        | `str`     | `'val_loss'`    | Metric to monitor (`'val_loss'` or `'val_accuracy'`) |
| `patience`                       | `int`     | `None`          | Early stopping patience: stop training after this many epochs with no improvement |
| `save_param_only`                | `bool`    | `False`         | Save only model parameters (exclude architecture and optimizer state) |
| `parallel_training_and_save`     | `bool`    | `False`         | Save checkpoints asynchronously in a background process (non-blocking) |
| `parallel_dump`                  | `bool`    | `False`         | **Parallel Parameter Dump**: Save each variable as a separate file in a folder (strongly recommended for very large models) |
| `callbacks`                      | `list`    | `[]`            | List of Keras-style callback objects |
| `end_loss` / `end_acc`           | `float`   | `None`          | Stop training when training loss/accuracy reaches this value |
| `end_test_loss` / `end_test_acc` | `float`   | `None`          | Stop training when validation loss/accuracy reaches this value |

---

# Examples

See the [examples directory](https://github.com/NoteDance/Note/tree/Note-7.0/Note/models/docs_example) for complete working examples:

- Basic training
- Distributed training (MirroredStrategy, MultiWorkerMirroredStrategy, ParameterServerStrategy)
- Fine-tuning pre-trained models
- Adaptive batch sizing
- Custom callbacks

---

# LRFinder:
**Usage:**

Create a Note model, then execute this code:
```python
from Note import nn
# model is a Note model
model.optimizer = tf.keras.optimizers.Adam()
lr_finder = nn.LRFinder(model)

# Train a model with batch size 512 for 5 epochs
# with learning rate growing exponentially from 0.0001 to 1
# N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]
lr_finder.find(N, train_ds, loss_object, train_loss, start_lr=0.0001, end_lr=1, batch_size=512, epochs=5)
```
or
```python
from Note import nn
# model is a Note model
model.optimizer = tf.keras.optimizers.Adam()
strategy = tf.distribute.MirroredStrategy()
lr_finder = nn.LRFinder(model)

# Train a model with batch size 512 for 5 epochs
# with learning rate growing exponentially from 0.0001 to 1
# N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]
lr_finder.find(N, train_ds, loss_object, strategy=strategy, start_lr=0.0001, end_lr=1, batch_size=512, epochs=5)
```
```python
# Plot the loss, ignore 20 batches in the beginning and 5 in the end
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
```
```python
# Plot rate of change of the loss
# Ignore 20 batches in the beginning and 5 in the end
# Smooth the curve using simple moving average of 20 batches
# Limit the range for y axis to (-0.02, 0.01)
lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
```

# OptFinder:
**Usage:**

Create a Note model, then execute this code:
```python
from Note import nn
# model is a Note model
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
opt_finder = nn.OptFinder(model, optimizers)

# Train a model with batch size 512 for 5 epochs
opt_finder.find(train_ds, loss_object, train_loss, batch_size=512)
```
or
```python
from Note import nn
# model is a Note model
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
strategy = tf.distribute.MirroredStrategy()
opt_finder = nn.OptFinder(model, optimizers)

# Train a model with batch size 512 for 5 epochs
opt_finder.find(train_ds, loss_object, strategy=strategy, batch_size=512)
```

# ParallelFinder:

**Overview**

The **ModelFinder** class is designed to help identify the best model during training by comparing losses across multiple models. It trains several models in parallel (using multiprocessing) and records the loss information at the end of each epoch. If the current epoch is the final one and the model’s loss is lower than the best recorded loss, the shared log is updated with the best optimizer and the lowest loss. This mechanism allows you to determine which model performed best after training.

This class supports two training modes:
- **Standard Training:** Invokes the model's `train` method.
- **Distributed Training:** When a distributed strategy is provided, it calls the model’s `distributed_training` method.

---

**Key Attributes**

- **models**  
  *Type:* `list`  
  *Description:* A list of model instances to be trained, each of which will run in its own process.

- **optimizers**  
  *Type:* `list`  
  *Description:* A list of optimizers corresponding to the models, which are used during the training process.

- **logs**  
  *Type:* Shared dictionary (created with `multiprocessing.Manager().dict()`)  
  *Description:* Records key information during training. Initially, it contains:
  - `best_loss`: Set to a large value (1e9) as a starting point for comparison.
  - Later, `best_opt` may be added to store the optimizer corresponding to the lowest loss.

- **lock**  
  *Type:* `multiprocessing.Lock`  
  *Description:* A multiprocessing lock to ensure safe access and modification of the shared `logs` dictionary among processes.

- **epochs**  
  *Type:* `int`  
  *Description:* The total number of training epochs, set in the `find` method. It is used to determine if the current epoch is the final one.

---

**Main Methods**

**1. `__init__(self, models, optimizers)`**

**Purpose:**  
Initializes a ModelFinder instance by setting the list of models and optimizers. It also creates a shared logs dictionary and a multiprocessing lock.

**Parameters:**
- `models`: A list of model instances.
- `optimizers`: A list of optimizers corresponding to the models.

**Details:**  
The constructor uses `multiprocessing.Manager` to create a shared `logs` dictionary, pre-setting `best_loss` to a high value (1e9) for later comparisons. A multiprocessing lock (`lock`) is created to ensure thread safety when multiple processes access the shared data.

**2. `on_epoch_end(self, epoch, logs, model=None, lock=None)`**

**Purpose:**  
Serves as a callback function executed at the end of each epoch. It checks whether the current epoch is the last one and, if so, updates the shared log with the best loss and corresponding optimizer.

**Parameters:**
- `epoch`: The current epoch number (starting from 0).
- `logs`: A dictionary containing training information for the current epoch, which must include the key `'loss'`.
- `model`: The model instance being trained (used to access the model's optimizer).
- `lock`: The multiprocessing lock used to synchronize access to the shared log.

**Key Logic:**
1. Acquire the lock using `lock.acquire()` to protect shared resources.
2. Retrieve the current loss from the `logs` dictionary.
3. Check if the current epoch is the final one (`epoch + 1 == self.epochs`).
4. If the current loss is lower than the previously recorded `best_loss`, update:
   - `logs['best_loss']` with the current loss.
   - `logs['best_opt']` with the model's optimizer.
5. Release the lock using `lock.release()`.

**3. `find(self, train_ds=None, loss_object=None, train_loss=None, strategy=None, batch_size=64, epochs=1, jit_compile=True)`**

**Purpose:**  
Starts the multiprocessing training of multiple models and uses a callback function to record the best loss and corresponding optimizer during training.

**Parameters:**
- `train_ds`: The training dataset.
- `loss_object`: The loss function used to compute training error.
- `train_loss`: The metric used to compute the training loss.
- `strategy`: The distributed training strategy (optional). If provided, the distributed training mode is used; otherwise, standard training is performed.
- `batch_size`: The batch size for training (default is 64).
- `epochs`: The total number of training epochs.
- `jit_compile`: Whether to enable JIT compilation for faster training (default is True).

**Key Logic:**
1. Store the passed `epochs` value in `self.epochs`.
2. Loop over each model and, for each:
   - Use `functools.partial` to create a `partial_callback` that binds the model, lock, and callback function to `epoch_end_callback`.
   - Create a callback instance using `nn.LambdaCallback` that triggers `on_epoch_end`.
   - Assign the corresponding optimizer to the model.
3. Depending on whether a `strategy` is provided, select the training method:
   - If `strategy` is `None`, use the model’s `train` method (standard training).
   - Otherwise, use the model’s `distributed_training` method (distributed training) with a lambda function that directly calls `on_epoch_end`.
4. For each model, start a new process with the corresponding training parameters (such as training dataset, loss function, number of epochs, callbacks, etc.).
5. Wait for all processes to finish by calling `join()` on each process.

---

**Example Usage**

Below is an example demonstrating how to use ModelFinder to train multiple models and select the best one based on the training loss.

```python
from Note import nn

# Assume model1 and model2 are properly initialized models, and optimizer1 and optimizer2 are their respective optimizers
model1 = ...  # Initialize model 1
model2 = ...  # Initialize model 2
optimizer1 = ...  # Optimizer for model 1
optimizer2 = ...  # Optimizer for model 2

# Create lists of models and optimizers
models = [model1, model2]
optimizers = [optimizer1, optimizer2]

# Initialize a ModelFinder instance
parallel_finder = nn.ParallelFinder(models, optimizers)

# Prepare training dataset and loss function (example)
train_dataset = ...  # Training dataset
loss_function = ...  # Loss function
train_loss_metric = ...  # Training loss metric

# Execute training in standard mode (without distributed strategy)
parallel_finder.find(
    train_ds=train_dataset,
    loss_object=loss_function,
    train_loss=train_loss_metric,
    epochs=10,
    jit_compile=True
)

# After training, the best result can be accessed via model_finder.logs
print("Best Loss:", model_finder.logs['best_loss'])
print("Best Loss Model:", model_finder.logs['best_loss_model'])
```
