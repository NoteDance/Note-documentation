# Introduction:
**Deep learning models built with Note are compatible with TensorFlow and can be trained with TensorFlow.**


# Train:
```python
from Note.neuralnetwork.tf.ConvNeXtV2 import ConvNeXtV2
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)
convnext_atto.build()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
      from_logits=True,
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
convnext_atto.fine_tuning(10)
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

ConvNeXt_tiny:
```python
from Note.neuralnetwork.tf.ConvNeXt import ConvNeXt
convnext_tiny=ConvNeXt(model_type='tiny',classes=1000)
convnext_tiny.build()
```

ConvNeXtV2_atto:
```python
from Note.neuralnetwork.tf.ConvNeXtV2 import ConvNeXtV2
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)
convnext_atto.build()
```

CLIP_large:
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

EfficientNetB0:
```python
from Note.neuralnetwork.tf.EfficientNet import EfficientNet
efficientnetb0=EfficientNet(model_name='B0',classes=1000)
efficientnetb0.build()
```

EfficientNetV2S:
```python
from Note.neuralnetwork.tf.EfficientNetV2 import EfficientNetV2
efficientnetv2s=EfficientNetV2(model_name='efficientnetv2-s',classes=1000)
efficientnetv2s.build()
```

Llama2_7B:
```python
from Note.neuralnetwork.tf.Llama2 import Llama2
llama=Llama2()
```

MobileNetV2:
```python
from Note.neuralnetwork.tf.MobileNetV2 import MobileNetV2
mobilenet=MobileNetV2(classes=1000)
mobilenet.build()
```

MobileNetV3_large:
```python
from Note.neuralnetwork.tf.MobileNetV3 import MobileNetV3
mobilenet=MobileNetV3(model_type="large",classes=1000)
mobilenet.build()
```

ResNet50:
```python
from Note.neuralnetwork.tf.ResNet.ResNet50 import ResNet50
resnet50=ResNet50(classes=1000)
resnet50.build()
```

ViT
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
    dropout=0.1,
    emb_dropout=0.1
)
```
