# Introduction:
**Deep learning models built with Note are compatible with TensorFlow and can be trained with TensorFlow.**


# Installation:
**Download Note from https://github.com/NoteDance/Note and then unzip it to the site-packages folder of your Python environment.**


# Train:
```python
from Note.neuralnetwork.tf.ConvNeXtV2 import ConvNeXtV2
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)
convnext_atto.build()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function(jit_compile=True)
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = convnext_atto(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, convnext_atto.param)
  optimizer.apply_gradients(zip(gradients, convnext_atto.param))
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


# Distributed training:
```python
from Note.neuralnetwork.tf.ConvNeXtV2 import ConvNeXtV2

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
  convnext_atto=ConvNeXtV2(model_type='atto',classes=10)
  convnext_atto.build()
  optimizer = tf.keras.optimizers.Adam()

def train_step(inputs):
  images, labels = inputs
  with tf.GradientTape() as tape:
    predictions = convnext_atto(images)
    loss = compute_loss(labels, predictions)
  gradients = tape.gradient(loss, convnext_atto.param)
  optimizer.apply_gradients(zip(gradients, convnext_atto.param))
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

  template = ("Epoch {}, Loss: {})
  print(template.format(epoch + 1, train_loss)
```


# Fine-tuning:
```python
convnext_atto.fine_tuning(10,flag=0)
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
convnext_atto.training=False
output=convnext_atto(data)
```


# Save model parameters:
```python
import pickle
output_file=open('param.dat','wb')
pickle.dump(convnext_atto.param,output_file)
output_file.close()
```


# Restore model parameters:
```python
import pickle
input_file=open('param.dat','rb')
param=pickle.load(input_file)
input_file.close()
```


# Assign the trained parameters to the model:
The assign_param function allows you to assign trained parameters, such as downloaded pre-trained parameters, to the parameters of a neural network. These parameters should be stored in a list.
```python
import pickle
from Note.neuralnetwork.tf.ConvNeXtV2 import ConvNeXtV2
from Note.neuralnetwork.tf.assign_param import assign_param
convnext_atto=ConvNeXtV2(model_type='atto',classes=10)
convnext_atto.build()
input_file=open('param.dat','rb')
param=pickle.load(input_file)
input_file.close()
assign_param(convnext_atto.param,param)
```


# Save model:
```python
import pickle
output_file=open('model.dat','wb')
pickle.dump(convnext_atto,output_file)
output_file.close()
```


# Restore model:
```python
import pickle
input_file=open('model.dat','rb')
convnext_atto=pickle.load(input_file)
input_file.close()
```


# Build models:
Here are some examples of building various neural networks, all in a similar way.

**ConvNeXt_tiny:**
```python
from Note.neuralnetwork.tf.ConvNeXt import ConvNeXt
convnext_tiny=ConvNeXt(model_type='tiny',classes=1000)
convnext_tiny.build()
```

**ConvNeXtV2_atto:**
```python
from Note.neuralnetwork.tf.ConvNeXtV2 import ConvNeXtV2
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)
convnext_atto.build()
```

**CLIP_large:**
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

**DiT_B_4:**
```python
from Note.neuralnetwork.tf.DiT import DiT_B_4
dit=DiT_B_4()
```

**EfficientNetB0:**
```python
from Note.neuralnetwork.tf.EfficientNet import EfficientNet
efficientnetb0=EfficientNet(model_name='B0',classes=1000)
efficientnetb0.build()
```

**EfficientNetV2S:**
```python
from Note.neuralnetwork.tf.EfficientNetV2 import EfficientNetV2
efficientnetv2s=EfficientNetV2(model_name='efficientnetv2-s',classes=1000)
efficientnetv2s.build()
```

**Llama2_7B:**
```python
from Note.neuralnetwork.tf.Llama2 import Llama2
llama=Llama2()
```

**MobileNetV2:**
```python
from Note.neuralnetwork.tf.MobileNetV2 import MobileNetV2
mobilenet=MobileNetV2(classes=1000)
mobilenet.build()
```

**MobileNetV3_large:**
```python
from Note.neuralnetwork.tf.MobileNetV3 import MobileNetV3
mobilenet=MobileNetV3(model_type="large",classes=1000)
mobilenet.build()
```

**ResNet50:**
```python
from Note.neuralnetwork.tf.ResNet.ResNet50 import ResNet50
resnet50=ResNet50(classes=1000)
resnet50.build()
```

**ViT**
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

**CaiT**
```python
import tensorflow as tf
from Note.neuralnetwork.tf.CaiT import CaiT

v = CaiT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 12,             # depth of transformer for patch to patch attention only
    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
    heads = 16,
    mlp_dim = 2048,
    dropout_rate = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05    # randomly dropout 5% of the layers
)

img = tf.random.normal((1, 256, 256, 3))

preds = v(img) # (1, 1000)
```

**PiT**
```python
import tensorflow as tf
from Note.neuralnetwork.tf.PiT import PiT

v = PiT(
    image_size = 224,
    patch_size = 14,
    dim = 256,
    num_classes = 1000,
    depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
    heads = 16,
    mlp_dim = 2048,
    dropout_rate = 0.1,
    emb_dropout = 0.1
)

# forward pass now returns predictions and the attention maps

img = tf.random.normal((1, 224, 224, 3))

preds = v(img) # (1, 1000)
```

**Cross ViT**
```python
import tensorflow as tf
from Note.neuralnetwork.tf.CrossViT import CrossViT

v = CrossViT(
    image_size = 256,
    num_classes = 1000,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
    lg_dim = 384,            # low res dimension
    lg_patch_size = 64,      # low res patch size
    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1
)

img = tf.random.normal((1, 256, 256, 3))

pred = v(img) # (1, 1000)
```

**Deep ViT**
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

preds = v(img) # (1, 1000)
```

**ViViT**
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

preds = v(video) # (4, 1000)
```

**XCiT**
```python
import tensorflow as tf
from Note.neuralnetwork.tf.XCiT import XCiT

v = XCiT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 12,                     # depth of xcit transformer
    cls_depth = 2,                  # depth of cross attention of CLS tokens to patch, attention pool at end
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05,           # randomly dropout 5% of the layers
    local_patch_kernel_size = 3     # kernel size of the local patch interaction module (depthwise convs)
)

img = tf.random.normal([1, 256, 256, 3])

preds = v(img) # (1, 1000)
```

**CvT**
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

pred = v(img) # (1, 1000)
```

**CCT**
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
pred = cct(img) # (1, 1000)
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

**MiT**
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

**BEiT**
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
