# Introduction:
Deep learning models built with Note are compatible with TensorFlow and can be trained with TensorFlow. The documentation shows how to train, test, save, and restore models built with Note.


# Train:
```python
import tensorflow as tf
from Note.neuralnetwork.docs_example.model1 import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model=Model()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function(jit_compile=True)
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.param)
  optimizer.apply_gradients(zip(gradients, model.param))
  train_loss(loss)

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
  )
```
or
```python
import tensorflow as tf
from Note.neuralnetwork.docs_example.model1 import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model=Model()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

model.fit(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If use early stopping.
# model.end_acc=0.9
# model.fit(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save the model at intervals of 1 epoch, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=1
# model. max_save_files=2
# model.fit(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save the model at intervals of 1875 batches, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq_=1875
# model. max_save_files=2
# model.fit(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save parameters only
# model.path='param.dat'
# model.save_freq=1
# model. max_save_files=2
# model.save_param_only=True
# model.fit(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.monitor='val_loss'
# model.fit(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy)

# If use parallel test(experiment)
# import multiprocessing as mp
# x_test, y_test = model.segment_data(x_test, y_test, 7)
# test_ds = [tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32) x_test,y_test for zip(x_test,y_test)]
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# test_loss = [tf.keras.metrics.Mean(name='test_loss') for _ in range(7)]
# test_accuracy = [tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy') for _ in range(7)]
# model.fit(train_ds, loss_object, train_loss, optimizer, 5, train_accuracy, test_ds, test_loss, test_accuracy, 7, mp)

# visualize
# model.visualize_train()
# model.visualize_test()
# model.visualize_comparison()

# save
# model.save_param('param.dat')
# model.save('model.dat')
```


# Distributed training:
```python
import tensorflow as tf
from Note.neuralnetwork.docs_example.model2 import Model

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
  model=Model()
  optimizer = tf.keras.optimizers.Adam()

def train_step(inputs):
  images, labels = inputs
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = compute_loss(labels, predictions)
  gradients = tape.gradient(loss, model.param)
  optimizer.apply_gradients(zip(gradients, model.param))
  return loss

@tf.function(jit_compile=True)
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

for epoch in range(EPOCHS):
  total_loss = 0.0
  num_batches = 0
  for x in train_dist_dataset:
    total_loss += distributed_train_step(x)
    num_batches += 1
  train_loss = total_loss / num_batches

  template = ("Epoch {}, Loss: {}")
  print(template.format(epoch + 1, train_loss)
```
or
```python
import tensorflow as tf
from Note.neuralnetwork.docs_example.model2 import Model

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)

with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

with strategy.scope():
  model=Model()
  optimizer = tf.keras.optimizers.Adam()

model.distributed_fit(train_dist_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
EPOCHS, train_accuracy, test_dist_dataset, test_loss, test_accuracy)

# If use early stopping.
# model.end_acc=0.9
# model.distributed_fit(train_dist_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy, test_dist_dataset, test_loss, test_accuracy)

# If save the model at intervals of 2 epoch, with a maximum of 3 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=2
# model.max_save_files=3
# model.distributed_fit(train_dist_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy, test_dist_dataset, test_loss, test_accuracy)

# If save the model at intervals of 1094 batches, with a maximum of 3 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq_=1094
# model.max_save_files=3
# model.distributed_fit(train_dist_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy, test_dist_dataset, test_loss, test_accuracy)

# If save parameters only
# model.path='param.dat'
# model.save_freq=2
# model.max_save_files=3
# model.save_param_only=True
# model.distributed_fit(train_dist_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy, test_dist_dataset, test_loss, test_accuracy)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.monitor='val_loss'
# model.distributed_fit(train_dist_dataset, loss_object, GLOBAL_BATCH_SIZE, optimizer, strategy,
# EPOCHS, train_accuracy, test_dist_dataset, test_loss, test_accuracy)

# visualize
# model.visualize_train()
# model.visualize_test()
# model.visualize_comparison()

# save
# model.save_param('param.dat')
# model.save('model.dat')
```


# Fine-tuning:
```python
model.fine_tuning(10,flag=0)
optimizer.lr=0.0001
fine_ds = tf.data.Dataset.from_tensor_slices((x_fine, y_fine)).batch(32)

EPOCHS = 1

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()

  for images, labels in fine_ds:
    train_step(images, labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
  )
```
flag=0:
Replace the pre-trained layer and assign the parameters of the fine-tuning layer to self.param.

flag=1:
Assign the parameters of the pre-trained layer and the parameters of the fine-tuning layer to self.param.

flag=2:
Restore the pre-trained layer and assign the parameters of the pre-trained layer to self.param.


# Use neural network:
```python
model.training()
output=model(data)
```


# Test model:
```python
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_loss, test_acc = model.test(test_ds, loss_object, test_loss, test_accuracy)
```
or parallel test
```python
import multiprocessing as mp
x_test, y_test = model.segment_data(x_test, y_test, 7)
test_ds = [tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32) x_test,y_test for zip(x_test,y_test)]
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
test_loss = [tf.keras.metrics.Mean(name='test_loss') for _ in range(7)]
test_accuracy = [tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy') for _ in range(7)]
test_loss, test_acc = model.test(test_ds, loss_object, test_loss, test_accuracy, 7, mp)
```


# Save model parameters:
```python
import pickle
output_file=open('param.dat','wb')
pickle.dump(model.param,output_file)
output_file.close()
```
or
```python
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
model.restore_param('param.dat')
```
or
```python
from Note import nn
param=nn.restore_param('param.dat')
```


# Assign the trained parameters to the model:
The assign_param function allows you to assign trained parameters, such as downloaded pre-trained parameters, to the parameters of a neural network. These parameters should be stored in a list.
```python
from Note import nn
from Note.neuralnetwork.tf.ConViT import convit_tiny
import pickle
model=convit_tiny(embed_dim=48)
input_file=open('param.dat','rb')
param=pickle.load(input_file)
input_file.close()
nn.assign_param(model.param,param)
```


# Save model:
```python
import pickle
output_file=open('model.dat','wb')
pickle.dump(model,output_file)
output_file.close()
```
or
```python
model.save('model.dat')
```


# Restore model:
```python
import pickle
input_file=open('model.dat','rb')
model=pickle.load(input_file)
input_file.close()
```
or
```python
model.restore('model.dat')
```
or
```python
from Note import nn
model,optimizer=nn.restore('model.dat')
```


# Build models:
If you want to create a new model without affecting the existing models, please use detach() to separate the current model’s variables from the global variables in the Model.

**separate the current model’s variables from the global variables in the Model**
```python
model.detach()
```

## ConvNeXt_tiny
```python
from Note.neuralnetwork.tf.ConvNeXt import ConvNeXt
convnext_tiny=ConvNeXt(model_type='tiny',classes=1000)
```

## ConvNeXtV2_atto
```python
from Note.neuralnetwork.tf.ConvNeXtV2 import ConvNeXtV2
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)
```

## CLIP_large
```python
from Note.neuralnetwork.tf.CLIP import CLIP
clip=CLIP(
    embed_dim=1024,
    image_resolution=224,
    vision_layers=14,
    vision_width=1024,
    vision_patch_size=32,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12
  )
```

## DiT_B_4
```python
from Note.neuralnetwork.tf.DiT import DiT_B_4
dit=DiT_B_4()
```

## EfficientNetB0
```python
from Note.neuralnetwork.tf.EfficientNet import EfficientNet
efficientnetb0=EfficientNet(model_name='B0',classes=1000)
```

## EfficientNetV2S
```python
from Note.neuralnetwork.tf.EfficientNetV2 import EfficientNetV2
efficientnetv2s=EfficientNetV2(model_name='efficientnetv2-s',classes=1000)
```

## Llama2_7B
```python
from Note.neuralnetwork.tf.Llama2 import Llama2
llama=Llama2()
```

## MobileNetV2
```python
from Note.neuralnetwork.tf.MobileNetV2 import MobileNetV2
mobilenet=MobileNetV2(classes=1000)
```

## MobileNetV3_large
```python
from Note.neuralnetwork.tf.MobileNetV3 import MobileNetV3
mobilenet=MobileNetV3(model_type="large",classes=1000)
```

## ResNet50
```python
from Note.neuralnetwork.tf.ResNet.ResNet50 import ResNet50
resnet50=ResNet50(classes=1000)
```

## ViT
```python
from Note.neuralnetwork.tf.ViT import ViT
vit=ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    pool='cls',
    channels=3,
    dim_head=64,
    drop_rate=0.1,
    emb_dropout=0.1
)
```

## CaiT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.CaiT import cait_XXS24_224

model = cait_XXS24_224()

img = tf.random.normal((1, 224, 224, 3))

output = model(img) # (1, 1000)
```

## PiT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.PiT import pit_b

model = pit_b()

# forward pass now returns predictions and the attention maps

img = tf.random.normal((1, 224, 224, 3))

output = model(img) # (1, 1000)
```

## Cross ViT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.CrossViT import crossvit_tiny_224()

model = crossvit_tiny_224()

img = tf.random.normal((1, 240, 240, 3))

output = model(img) # (1, 1000)
```

## Deep ViT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.DeepViT import DeepViT

v = DeepViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout_rate = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal((1, 256, 256, 3))

output = v(img) # (1, 1000)
```

## ViViT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.ViViT import ViViT

v = ViViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    spatial_depth = 6,         # depth of the spatial transformer
    temporal_depth = 6,        # depth of the temporal transformer
    heads = 8,
    mlp_dim = 2048
)

video = tf.random.normal((4, 16, 128, 128, 3)) # (batch, frames, height, width, channels)

output = v(video) # (4, 1000)
```

## XCiT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.XCiT import xcit_nano_12_p16

model = xcit_nano_12_p16()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```

## CvT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.CvT import CvT

v = CvT(
    num_classes = 1000,
    s1_emb_dim = 64,        # stage 1 - dimension
    s1_emb_kernel = 7,      # stage 1 - conv kernel
    s1_emb_stride = 4,      # stage 1 - conv stride
    s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
    s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
    s1_heads = 1,           # stage 1 - heads
    s1_depth = 1,           # stage 1 - depth
    s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
    s2_emb_dim = 192,       # stage 2 - (same as above)
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,       # stage 3 - (same as above)
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 4,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.
)

img = tf.random.normal((1, 224, 224, 3))

output = v(img) # (1, 1000)
```

## CCT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.CCT import CCT

cct = CCT(
    img_size = (224, 448),
    embedding_dim = 384,
    n_conv_layers = 2,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 14,
    num_heads = 6,
    mlp_ratio = 3.,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)

img = tf.random.normal((1, 224, 448, 3))
output = cct(img) # (1, 1000)
```
Alternatively you can use one of several pre-defined models [2,4,6,7,8,14,16] which pre-define the number of layers, number of attention heads, the mlp ratio, and the embedding dimension.
```python
from Note.neuralnetwork.tf.CCT import cct_14

cct = cct_14(
    img_size = 224,
    n_conv_layers = 1,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)
```

## MiT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.MiT import mit_b0
model = mit_b0()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size, in_chans])

output = model(img)
```

## BEiT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.BEiT import beit_base_patch16_224
model = beit_base_patch16_224()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## SwinMLP
```python
import tensorflow as tf
from Note.neuralnetwork.tf.SwinMLP import SwinMLP
model = SwinMLP()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## SwinTransformerV2
```python
import tensorflow as tf
from Note.neuralnetwork.tf.SwinTransformerV2 import SwinTransformerV2
model = SwinTransformerV2()

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## ConViT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.ConViT import convit_tiny
model = convit_tiny(embed_dim=48)

batch_size = 10
img_size = 224
in_chans = 3
img = tf.random.normal([batch_size, img_size, img_size,in_chans])

output = model(img)
```

## PVT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.PVT import pvt_v2_b0

model = pvt_v2_b0()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```

## GCViT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.GCViT import gc_vit_xxtiny

model = gc_vit_xxtiny()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```

## DaViT
```python
import tensorflow as tf
from Note.neuralnetwork.tf.DaViT import davit_tiny

model = davit_tiny()

img = tf.random.normal([1, 224, 224, 3])

output = model(img) # (1, 1000)
```
