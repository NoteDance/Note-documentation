# adaptive_avg_pooling1d

The `adaptive_avg_pooling1d` class implements a 1D adaptive average pooling layer. This layer reduces the input tensor along the specified dimension to a new size defined by `output_size`.

**Initialization Parameters**

- **`output_size`** (int or iterable of int): Specifies the desired size of the output features. This can be an integer or a list/tuple of a single integer.
- **`data_format`** (str, default='channels_last'): Specifies the ordering of the dimensions in the input data. `channels_last` corresponds to inputs with shape `(batch, steps, features)` while `channels_first` corresponds to inputs with shape `(batch, features, steps)`.

**Methods**

- **`__call__(self, data)`**: Applies the adaptive average pooling operation to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor to be pooled.
  
  - **Returns**:
    - **`out_vect`** (tensor): Output tensor after adaptive average pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of adaptive_avg_pooling1d
pooling_layer = adaptive_avg_pooling1d(output_size=4)

# Generate some sample data
data = tf.random.normal((32, 16, 8))  # Batch of 32 samples, 16 timesteps, 8 features

# Apply adaptive average pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 4, 8) for 'channels_last' data format
```

# adaptive_avg_pooling2d

The `adaptive_avg_pooling2d` class implements a 2D adaptive average pooling layer. This layer reduces the input tensor along the specified dimensions to a new size defined by `output_size`.

**Initialization Parameters**

- **`output_size`** (int or iterable of int): Specifies the desired size of the output features as (pooled_rows, pooled_cols). This can be an integer or a list/tuple of two integers.
- **`data_format`** (str, default='channels_last'): Specifies the ordering of the dimensions in the input data. `channels_last` corresponds to inputs with shape `(batch, height, width, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, height, width)`.

**Methods**

- **`__call__(self, data)`**: Applies the adaptive average pooling operation to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor to be pooled.
  
  - **Returns**:
    - **`out_vect`** (tensor): Output tensor after adaptive average pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of adaptive_avg_pooling2d
pooling_layer = nn.adaptive_avg_pooling2d(output_size=(4, 4))

# Generate some sample data
data = tf.random.normal((32, 16, 16, 8))  # Batch of 32 samples, 16x16 spatial dimensions, 8 channels

# Apply adaptive average pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 4, 4, 8) for 'channels_last' data format
```

# adaptive_avg_pooling3d

The `adaptive_avg_pooling3d` class implements a 3D adaptive average pooling layer. This layer reduces the input tensor along the specified dimensions to a new size defined by `output_size`.

**Initialization Parameters**

- **`output_size`** (int or iterable of int): Specifies the desired size of the output features as (pooled_dim1, pooled_dim2, pooled_dim3). This can be an integer or a list/tuple of three integers.
- **`data_format`** (str, default='channels_last'): Specifies the ordering of the dimensions in the input data. `channels_last` corresponds to inputs with shape `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.

**Methods**

- **`__call__(self, data)`**: Applies the adaptive average pooling operation to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor to be pooled.
  
  - **Returns**:
    - **`out_vect`** (tensor): Output tensor after adaptive average pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of adaptive_avg_pooling3d
pooling_layer = nn.adaptive_avg_pooling3d(output_size=(4, 4, 4))

# Generate some sample data
data = tf.random.normal((32, 16, 16, 16, 8))  # Batch of 32 samples, 16x16x16 spatial dimensions, 8 channels

# Apply adaptive average pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 4, 4, 4, 8) for 'channels_last' data format
```

# adaptive_max_pooling1d

The `adaptive_max_pooling1d` class implements a 1D adaptive max pooling layer. This layer reduces the input tensor along the specified dimension to a new size defined by `output_size`.

**Initialization Parameters**

- **`output_size`** (int or iterable of int): Specifies the desired size of the output features as a single integer. This can be an integer or a list/tuple containing a single integer.
- **`data_format`** (str, default='channels_last'): Specifies the ordering of the dimensions in the input data. `channels_last` corresponds to inputs with shape `(batch, steps, features)` while `channels_first` corresponds to inputs with shape `(batch, features, steps)`.

**Methods**

- **`__call__(self, data)`**: Applies the adaptive max pooling operation to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor to be pooled.
  
  - **Returns**:
    - **`out_vect`** (tensor): Output tensor after adaptive max pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of adaptive_max_pooling1d
pooling_layer = nn.adaptive_max_pooling1d(output_size=4)

# Generate some sample data
data = tf.random.normal((32, 16, 8))  # Batch of 32 samples, 16 steps, 8 features

# Apply adaptive max pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 4, 8) for 'channels_last' data format
```

# adaptive_max_pooling2d

The `adaptive_max_pooling2d` class implements a 2D adaptive max pooling layer. This layer reduces the input tensor along the specified dimensions to a new size defined by `output_size`.

**Initialization Parameters**

- **`output_size`** (int or iterable of int): Specifies the desired size of the output features as two integers, representing the number of pooled rows and columns. This can be an integer or a list/tuple containing two integers.
- **`data_format`** (str, default='channels_last'): Specifies the ordering of the dimensions in the input data. `channels_last` corresponds to inputs with shape `(batch, height, width, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, height, width)`.

**Methods**

- **`__call__(self, data)`**: Applies the adaptive max pooling operation to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor to be pooled.
  
  - **Returns**:
    - **`out_vect`** (tensor): Output tensor after adaptive max pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of adaptive_max_pooling2d
pooling_layer = adaptive_max_pooling2d(output_size=(4, 4))

# Generate some sample data
data = tf.random.normal((32, 16, 16, 8))  # Batch of 32 samples, 16x16 spatial dimensions, 8 channels

# Apply adaptive max pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 4, 4, 8) for 'channels_last' data format
```

# adaptive_max_pooling3d

The `adaptive_max_pooling3d` class implements a 3D adaptive max pooling layer. This layer reduces the input tensor along the specified dimensions to a new size defined by `output_size`.

**Initialization Parameters**

- **`output_size`** (int or iterable of int): Specifies the desired size of the output features as three integers, representing the number of pooled dimensions. This can be an integer or a list/tuple containing three integers.
- **`data_format`** (str, default='channels_last'): Specifies the ordering of the dimensions in the input data. `channels_last` corresponds to inputs with shape `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first` corresponds to inputs with shape `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.

**Methods**

- **`__call__(self, data)`**: Applies the adaptive max pooling operation to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor to be pooled.
  
  - **Returns**:
    - **`out_vect`** (tensor): Output tensor after adaptive max pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of adaptive_max_pooling3d
pooling_layer = nn.adaptive_max_pooling3d(output_size=(4, 4, 4))

# Generate some sample data
data = tf.random.normal((32, 16, 16, 16, 8))  # Batch of 32 samples, 16x16x16 spatial dimensions, 8 channels

# Apply adaptive max pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 4, 4, 4, 8) for 'channels_last' data format
```

# additive_attention

The `additive_attention` class implements an additive attention mechanism, which calculates attention scores as a nonlinear sum of query and key tensors.

**Initialization Parameters**

- **`input_size`** (int, optional): The size of the input tensor. If provided, it is used to initialize the scale parameter.
- **`use_scale`** (bool, default=True): Whether to use a learnable scale parameter.
- **`dtype`** (str, default='float32'): The data type of the input and scale parameter.

**Methods**

- **`__call__(self, query, key)`**: Computes the attention scores.

  - **Parameters**:
    - **`query`** (tensor): Query tensor of shape `[batch_size, Tq, dim]`.
    - **`key`** (tensor): Key tensor of shape `[batch_size, Tv, dim]`.
  
  - **Returns**:
    - **`Tensor`**: Attention scores of shape `[batch_size, Tq, Tv]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of additive_attention
attention_layer = nn.additive_attention(input_size=128)

# Generate some sample data
query = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 query steps, 128 dimensions
key = tf.random.normal((32, 20, 128))    # Batch of 32 samples, 20 key steps, 128 dimensions

# Apply attention
output = attention_layer(query, key)

print(output.shape)  # Output shape will be (32, 10, 20)
```

# avg_pool1d

The `avg_pool1d` class performs 1D average pooling on the input tensor.

**Initialization Parameters**

- **`ksize`** (int): Size of the window for each dimension of the input tensor.
- **`strides`** (int): Stride of the sliding window for each dimension of the input tensor.
- **`padding`** (str): Padding algorithm to use ('SAME' or 'VALID').

**Methods**

- **`__call__(self, data)`**: Applies 1D average pooling to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, length, channels]`.
  
  - **Returns**:
    - **`Tensor`**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of avg_pool1d
pooling_layer = nn.avg_pool1d(ksize=2, strides=2, padding='SAME')

# Generate some sample data
data = tf.random.normal((32, 100, 64))  # Batch of 32 samples, 100 steps, 64 channels

# Apply 1D average pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 50, 64)
```

# avg_pool2d

The `avg_pool2d` class performs 2D average pooling on the input tensor.

**Initialization Parameters**

- **`ksize`** (int or tuple of 2 ints): Size of the window for each dimension of the input tensor.
- **`strides`** (int or tuple of 2 ints): Stride of the sliding window for each dimension of the input tensor.
- **`padding`** (str): Padding algorithm to use ('SAME' or 'VALID').

**Methods**

- **`__call__(self, data)`**: Applies 2D average pooling to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, height, width, channels]`.
  
  - **Returns**:
    - **`Tensor`**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of avg_pool2d
pooling_layer = nn.avg_pool2d(ksize=(2, 2), strides=(2, 2), padding='SAME')

# Generate some sample data
data = tf.random.normal((32, 64, 64, 3))  # Batch of 32 samples, 64x64 spatial dimensions, 3 channels

# Apply 2D average pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will be (32, 32, 32, 3)
```

# avg_pool3d

The `avg_pool3d` class performs 3D average pooling on the input tensor.

**Initialization Parameters**

- **`ksize`** (int or tuple of 3 ints): Size of the window for each dimension of the input tensor.
- **`strides`** (int or tuple of 3 ints): Stride of the sliding window for each dimension of the input tensor.
- **`padding`** (str): Padding algorithm to use ('SAME' or 'VALID').

**Methods**

- **`__call__(self, data)`**: Applies 3D average pooling to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, depth, height, width, channels]`.
  
  - **Returns**:
    - **`Tensor`**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of avg_pool3d
pooling_layer = nn.avg_pool3d(ksize=(2, 2, 2), strides=(2, 2, 2), padding='SAME')

# Generate some sample data
data = tf.random.normal((16, 32, 32, 32, 3))  # Batch of 16 samples, 32x32x32 spatial dimensions, 3 channels

# Apply 3D average pooling
output = pooling_layer(data)

print(output.shape)  # Output shape will depend on the input shape, ksize, strides, and padding
```

# axial_positional_encoding

The `axial_positional_encoding` class generates axial positional encodings for Reformer models.

**Initialization Parameters**

- **`d_model`** (int): The dimension of the model embeddings.
- **`axial_shape`** (tuple of int): The shape of the input sequence, such as `(batch_size, seq_length)`.
- **`initializer`** (str): The initializer to use for the positional encoding weights (default is 'Xavier').
- **`trainable`** (bool): Whether the positional encodings are trainable (default is `True`).
- **`dtype`** (str): The data type of the positional encodings (default is 'float32').

**Methods**

- **`__call__(self, data)`**: Generates the axial positional encoding for the input tensor.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, seq_length, d_model]`.
  
  - **Returns**:
    - **`Tensor`**: Output tensor with axial positional encoding added.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of axial_positional_encoding
axial_pe = nn.axial_positional_encoding(d_model=512, axial_shape=(32, 128))

# Generate some sample data
data = tf.random.normal((32, 128, 512))  # Batch of 32 samples, 128 sequence length, 512 dimensions

# Apply axial positional encoding
output = axial_pe(data)

print(output.shape)  # Output shape will be (32, 128, 512)
```

# attention

This class implements an attention mechanism for neural networks, supporting both dot-product and concatenation-based attention scoring methods. It also allows for optional scaling of attention scores.

**Initialization Parameters**

- **`use_scale`** (bool): If `True`, scales the attention scores. Default is `False`.
- **`score_mode`** (str): The method to calculate attention scores. Options are `"dot"` (default) and `"concat"`.
- **`dtype`** (str): The data type for computations. Default is `'float32'`.

**Methods**

- **`__call__(self, query, value, key=None)`**: Applies the attention mechanism to the provided tensors.

  - **Parameters**:
    - **`query`** (Tensor): The query tensor.
    - **`value`** (Tensor): The value tensor.
    - **`key`** (Tensor, optional): The key tensor. If not provided, `value` is used as the key.

  - **Returns**:
    - **`Tensor`**: The result of the attention mechanism applied to the input tensors.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the attention class
att = nn.attention(use_scale=True, score_mode="dot", dtype='float32')

# Define sample query and value tensors
query = tf.random.normal(shape=(2, 5, 10))  # (batch_size, query_length, dim)
value = tf.random.normal(shape=(2, 6, 10))  # (batch_size, value_length, dim)

# Compute attention output
output = att(query, value)

print(output.shape)  # Should be (2, 5, 10)
```

# batch_norm

The `batch_norm` class implements batch normalization, which helps to stabilize and accelerate training by normalizing the input layer by adjusting and scaling the activations.

**Initialization Parameters**

- **`input_size`** (int, optional): Size of the input.
- **`axis`** (int): Axis along which to normalize. Default is `-1`.
- **`momentum`** (float): Momentum for the moving average. Default is `0.99`.
- **`epsilon`** (float): Small constant to avoid division by zero. Default is `0.001`.
- **`center`** (bool): If `True`, add offset of `beta` to the normalized tensor. Default is `True`.
- **`scale`** (bool): If `True`, multiply by `gamma`. Default is `True`.
- **`beta_initializer`** (str, list, tuple): Initializer for the beta weight. Default is `'zeros'`.
- **`gamma_initializer`** (str, list, tuple): Initializer for the gamma weight. Default is `'ones'`.
- **`moving_mean_initializer`** (str, list, tuple): Initializer for the moving mean. Default is `'zeros'`.
- **`moving_variance_initializer`** (str, list, tuple): Initializer for the moving variance. Default is `'ones'`.
- **`synchronized`** (bool): If `True`, synchronize the moments across replicas. Default is `False`.
- **`trainable`** (bool): If `True`, add variables to the trainable variables collection. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data, train_flag=None, mask=None)`**: Applies batch normalization to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`train_flag`** (bool, optional): Specifies whether the layer is in training mode.
    - **`mask`** (tensor, optional): Mask tensor for weighted moments calculation.

  - **Returns**: Normalized output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the batch normalization layer
bn = nn.batch_norm(input_size=10)

# Generate some sample data
data = tf.random.normal((2, 5, 10))

# Apply batch normalization
output = bn(data)
```

# cached_attention

The `cached_attention` class implements an attention mechanism with caching, primarily used for autoregressive decoding.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`key_dim`** (int): Dimension of the keys.
- **`value_dim`** (int, optional): Dimension of the values. Defaults to the same as `key_dim` if not specified.
- **`input_size`** (int, optional): Size of the input. If not specified, it will be inferred from the input data.
- **`attention_axes`** (list or tuple of ints, optional): Axes along which to apply attention. Defaults to the last axis.
- **`dropout_rate`** (float, optional): Dropout rate to apply to the attention scores. Defaults to 0.0.
- **`weight_initializer`** (str, list, tuple): Initializer for the weights. Defaults to "Xavier".
- **`bias_initializer`** (str, list, tuple): Initializer for the biases. Defaults to "zeros".
- **`use_bias`** (bool, optional): Whether to use bias in the dense layers. Defaults to True.
- **`dtype`** (str, optional): Data type of the layer. Defaults to 'float32'.

**Methods**

- **`build(self)`**: Builds the internal dense layers if `input_size` was not provided during initialization.

- **`_masked_softmax(self, attention_scores, attention_mask=None)`**: Applies a softmax operation to the attention scores with an optional mask.
  - **Parameters**:
    - **`attention_scores`** (tensor): Raw attention scores.
    - **`attention_mask`** (tensor, optional): Mask to apply to the attention scores.

  - **Returns**:
    - **`Tensor`**: Normalized attention scores.

- **`_update_cache(self, key, value, cache, decode_loop_step)`**: Updates the cache with new keys and values during decoding.
  - **Parameters**:
    - **`key`** (tensor): New keys.
    - **`value`** (tensor): New values.
    - **`cache`** (dict): Cache containing previous keys and values.
    - **`decode_loop_step`** (int, optional): Current step in the decoding loop.

  - **Returns**:
    - **`Tensor`**: Updated keys.
    - **`Tensor`**: Updated values.

- **`__call__(self, query, value, key=None, attention_mask=None, cache=None, decode_loop_step=None, return_attention_scores=False)`**: Computes the attention output and optionally returns the attention scores and updated cache.
  - **Parameters**:
    - **`query`** (tensor): Query tensor.
    - **`value`** (tensor): Value tensor.
    - **`key`** (tensor, optional): Key tensor. If not provided, the value tensor will be used.
    - **`attention_mask`** (tensor, optional): Mask to apply to the attention scores.
    - **`cache`** (dict, optional): Cache for storing previous keys and values.
    - **`decode_loop_step`** (int, optional): Current step in the decoding loop.
    - **`return_attention_scores`** (bool, optional): Whether to return the attention scores.

  - **Returns**:
    - **`Tensor`**: Attention output.
    - **`dict`**: Updated cache.
    - **`Tensor`** (optional): Attention scores if `return_attention_scores` is True.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of cached_attention
attention_layer = nn.cached_attention(
    n_head=8,
    key_dim=64,
    input_size=128,
    attention_axes=[1],
    dropout_rate=0.1
)

# Generate some sample data
query = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 sequence length, 128 input size
value = tf.random.normal((32, 10, 128))

# Apply the cached attention layer
output, cache = attention_layer(query, value)

print(output.shape)  # Output shape will be (32, 10, 128)
```

# capsule

This class implements a Capsule layer for neural networks, supporting both fully connected (FC) and convolutional (CONV) capsule layers with routing mechanisms.

**Initialization Parameters**

- **`num_outputs`** (int): The number of output capsules in this layer.
- **`vec_len`** (int): The length of the output vector of a capsule.
- **`input_shape`** (tuple, optional): The shape of the input tensor. Required for layer building.
- **`kernel_size`** (int, optional): The kernel size for convolutional capsule layers.
- **`stride`** (int, optional): The stride for convolutional capsule layers.
- **`with_routing`** (bool): Whether this capsule layer uses routing with the lower-level capsules. Default is `True`.
- **`layer_type`** (str): The type of capsule layer. Options are `'FC'` for fully connected or `'CONV'` for convolutional. Default is `'FC'`.
- **`iter_routing`** (int): The number of routing iterations. Default is `3`.
- **`steddev`** (float): The standard deviation for initializing the weights. Default is `0.01`.

**Methods**

- **`build(self)`**: Builds the layer based on the type and input shape. Should be called if `input_shape` is provided during initialization.

- **`__call__(self, data)`**: Applies the capsule layer to the provided input tensor.

  - **Parameters**:
    - **`data`** (Tensor): The input tensor.

  - **Returns**:
    - **`Tensor`**: The output tensor after applying the capsule layer.

- **`routing(self, input, b_IJ, num_outputs=10, num_dims=16)`**: The routing algorithm for the capsule layer.

  - **Parameters**:
    - **`input`** (Tensor): Input tensor with shape `[batch_size, num_caps_l, 1, length(u_i), 1]`.
    - **`b_IJ`** (Tensor): Initial logits for routing.
    - **`num_outputs`** (int): Number of output capsules.
    - **`num_dims`** (int): Number of dimensions for output capsule.

  - **Returns**:
    - **`Tensor`**: The output tensor after applying routing.

- **`squash(self, vector)`**: Squashing function to ensure that the length of the output vector is between 0 and 1.

  - **Parameters**:
    - **`vector`** (Tensor): Input tensor to be squashed.

  - **Returns**:
    - **`Tensor`**: Squashed output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Define input tensor
input_tensor = tf.random.normal(shape=(32, 28, 28, 256))  # Example shape

# Instantiate the capsule class
caps_layer = nn.capsule(num_outputs=10, vec_len=16, input_shape=input_tensor.shape, layer_type='FC', with_routing=True)

# Apply the capsule layer to the input tensor
output = caps_layer(input_tensor)

print(output.shape)  # Should be (32, 10, 16, 1) if the num_outputs is 10 and vec_len is 16
```

# ClassifierHead

This class implements a classifier head with configurable global pooling and dropout options.

**Initialization Parameters**

- **`in_features`** (int): The number of input features.
- **`num_classes`** (int): The number of classes for the final classifier layer (output).
- **`pool_type`** (str, optional): Type of global pooling. Options are `'avg'`, `'max'`, or `''` (no pooling). Default is `'avg'`.
- **`drop_rate`** (float, optional): Dropout rate before the classifier. Default is `0.`.
- **`use_conv`** (bool, optional): Whether to use convolution for the classifier. Default is `False`.
- **`input_fmt`** (str, optional): The input format. Options are `'NHWC'` or `'NCHW'`. Default is `'NHWC'`.

**Methods**

- **`reset(self, num_classes, pool_type=None)`**: Resets the classifier head with a new number of classes and optionally a new pooling type.

  - **Parameters**:
    - **`num_classes`** (int): New number of output classes.
    - **`pool_type`** (str, optional): New pooling type.

**`__call__(self, x, pre_logits=False)`**: Applies the classifier head to the provided input tensor.

  - **Parameters**:
    - **`x`** (Tensor): Input tensor.
    - **`pre_logits`** (bool, optional): Whether to return the features before the final logits layer. Default is `False`.

  - **Returns**:
    - **`Tensor`**: The output tensor after applying the classifier head.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Define input tensor
input_tensor = tf.random.normal(shape=(32, 7, 7, 2048))  # Example shape

# Instantiate the classifier head
classifier_head = nn.ClassifierHead(in_features=2048, num_classes=1000, pool_type='avg', drop_rate=0.5)

# Apply the classifier head to the input tensor
output = classifier_head(input_tensor)

print(output.shape)  # Should be (32, 1000) if num_classes is 1000
```

# NormMlpClassifierHead

This class implements a classifier head with normalization, configurable MLP, and global pooling options.

**Initialization Parameters**

- `in_features` (int): The number of input features.
- `num_classes` (int): The number of classes for the final classifier layer (output).
- `hidden_size` (int, optional): The hidden size of the MLP (pre-logits FC layer). Default is `None`.
- `pool_type` (str, optional): Type of global pooling. Options are `'avg'`, `'max'`, or `''` (no pooling). Default is `'avg'`.
- `drop_rate` (float, optional): Dropout rate before the classifier. Default is `0.`.
- `norm_layer` (Callable, optional): Normalization layer type. Default is `layer_norm`.
- `act_layer` (Callable, optional): Activation layer type. Default is `tf.nn.tanh`.

**Methods**

**`reset(self, num_classes, pool_type=None)`**: Resets the classifier head with a new number of classes and optionally a new pooling type.

  - **Parameters**:
    - **`num_classes`** (int): New number of output classes.
    - **`pool_type`** (str, optional): New pooling type.

**`__call__(self, x, pre_logits=False)`**: Applies the normalized MLP classifier head to the provided input tensor.

  - **Parameters**:
    - **`x`** (Tensor): Input tensor.
    - **`pre_logits`** (bool, optional): Whether to return the features before the final logits layer. Default is `False`.

  - Returns:
    - **`Tensor`**: The output tensor after applying the classifier head.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Define input tensor
input_tensor = tf.random.normal(shape=(32, 7, 7, 2048))  # Example shape

# Instantiate the normalized MLP classifier head
norm_mlp_head = nn.NormMlpClassifierHead(in_features=2048, num_classes=1000, hidden_size=512, pool_type='avg', drop_rate=0.5)

# Apply the normalized MLP classifier head to the input tensor
output = norm_mlp_head(input_tensor)

print(output.shape)  # Should be (32, 1000) if num_classes is 1000
```

# conv1d

The `conv1d` class implements a 1D convolutional layer, which is commonly used in processing sequential data such as time series or audio.

**Initialization Parameters**

- **`filters`** (int): Number of output filters in the convolution.
- **`kernel_size`** (int or list of int): Size of the convolutional kernel.
- **`input_size`** (int, optional): Size of the input channels.
- **`strides`** (int or list of int): Stride size for the convolution. Default is `[1]`.
- **`padding`** (str or list of int): Padding type or size. Default is `'VALID'`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format, either `'NWC'` or `'NCW'`. Default is `'NWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for dilated convolution. Default is `None`.
- **`groups`** (int): Number of groups for grouped convolution. Default is `1`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the 1D convolution to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 1D convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the conv1d layer
conv_layer = nn.conv1d(filters=32, kernel_size=3, input_size=64, strides=1, padding='SAME', activation='relu')

# Generate some sample data
data = tf.random.normal((10, 100, 64))

# Apply the convolutional layer
output = conv_layer(data)

print(output.shape)  # Output shape will be (10, 100, 32) if padding is 'SAME'
```

# conv1d_transpose

The `conv1d_transpose` class implements a 1D transposed convolutional layer, often used for tasks like upsampling in sequence data.

**Initialization Parameters**

- **`filters`** (int): Number of output filters in the transposed convolution.
- **`kernel_size`** (int or list of int): Size of the convolutional kernel.
- **`input_size`** (int, optional): Size of the input channels.
- **`strides`** (int or list of int): Stride size for the transposed convolution. Default is `[1]`.
- **`padding`** (str): Padding type. Default is `'VALID'`.
- **`output_padding`** (int, optional): Additional size added to the output shape.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format, either `'NWC'` or `'NCW'`. Default is `'NWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for dilated convolution. Default is `None`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the 1D transposed convolution to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 1D transposed convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the conv1d_transpose layer
conv_transpose_layer = nn.conv1d_transpose(filters=32, kernel_size=3, input_size=64, strides=[1], padding='SAME', activation='relu')

# Generate some sample data
data = tf.random.normal((10, 100, 64))

# Apply the transposed convolutional layer
output = conv_transpose_layer(data)

print(output.shape)  # Output shape will depend on strides and padding
```

# conv2d

The `conv2d` class implements a 2D convolutional layer, which is commonly used in image processing tasks.

**Initialization Parameters**

- **`filters`** (int): Number of output filters in the convolution.
- **`kernel_size`** (int or list of int): Size of the convolutional kernel. If a single integer is provided, it is used for both dimensions.
- **`input_size`** (int, optional): Number of input channels. If not provided, it will be inferred from the input data.
- **`strides`** (int or list of int): Stride size for the convolution. Default is `[1, 1]`.
- **`padding`** (str or list of int): Padding type or size. Default is `'VALID'`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format, either `'NHWC'` or `'NCHW'`. Default is `'NHWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for dilated convolution. Default is `None`.
- **`groups`** (int): Number of groups for group convolution. Default is `1`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the 2D convolution to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 2D convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the conv2d layer
conv_layer = nn.conv2d(filters=32, kernel_size=3, input_size=64, strides=2, padding='SAME', activation='relu')

# Generate some sample data
data = tf.random.normal((10, 128, 128, 64))  # Batch of 10 images, 128x128 pixels, 64 channels

# Apply the convolutional layer
output = conv_layer(data)

print(output.shape)  # Output shape will depend on strides and padding
```

# conv2d_transpose

The `conv2d_transpose` class implements a 2D transposed convolutional layer, which is commonly used for upsampling in image processing tasks.

**Initialization Parameters**

- **`filters`** (int): Number of output filters in the transposed convolution.
- **`kernel_size`** (int or list of int): Size of the transposed convolutional kernel. If a single integer is provided, it is used for both dimensions.
- **`input_size`** (int, optional): Number of input channels. If not provided, it will be inferred from the input data.
- **`strides`** (int or list of int): Stride size for the transposed convolution. Default is `[1, 1]`.
- **`padding`** (str): Padding type, either `'VALID'` or `'SAME'`. Default is `'VALID'`.
- **`output_padding`** (int or list of int, optional): Additional size added to one side of each dimension in the output shape. Default is `None`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format, either `'NHWC'` or `'NCHW'`. Default is `'NHWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for dilated transposed convolution. Default is `None`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the 2D transposed convolution to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 2D transposed convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the conv2d_transpose layer
conv_transpose_layer = nn.conv2d_transpose(filters=32, kernel_size=3, input_size=64, strides=2, padding='SAME', activation='relu')

# Generate some sample data
data = tf.random.normal((10, 64, 64, 64))  # Batch of 10 images, 64x64 pixels, 64 channels

# Apply the transposed convolutional layer
output = conv_transpose_layer(data)

print(output.shape)  # Output shape will depend on strides and padding
```

# conv3d

The `conv3d` class implements a 3D convolutional layer, which is commonly used for volumetric data such as videos or 3D medical images.

**Initialization Parameters**

- **`filters`** (int): Number of output filters in the convolution.
- **`kernel_size`** (int or list of int): Size of the convolutional kernel. If a single integer is provided, it is used for all three dimensions.
- **`input_size`** (int, optional): Number of input channels. If not provided, it will be inferred from the input data.
- **`strides`** (int or list of int): Stride size for the convolution. Default is `[1, 1, 1]`.
- **`padding`** (str): Padding type, either `'VALID'` or `'SAME'`. Default is `'VALID'`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format, either `'NDHWC'` or `'NCDHW'`. Default is `'NDHWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for dilated convolution. Default is `None`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the 3D convolution to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 3D convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the conv3d layer
conv_layer = nn.conv3d(filters=32, kernel_size=3, input_size=64, strides=2, padding='SAME', activation='relu')

# Generate some sample data
data = tf.random.normal((10, 64, 64, 64, 64))  # Batch of 10 volumetric data, 64x64x64 voxels, 64 channels

# Apply the convolutional layer
output = conv_layer(data)

print(output.shape)  # Output shape will depend on strides and padding
```

# conv3d_transpose

The `conv3d_transpose` class implements a 3D transposed convolutional (also known as deconvolutional) layer, which is commonly used for upsampling volumetric data such as videos or 3D medical images.

**Initialization Parameters**

- **`filters`** (int): Number of output filters in the transposed convolution.
- **`kernel_size`** (int or list of int): Size of the transposed convolutional kernel. If a single integer is provided, it is used for all three dimensions.
- **`input_size`** (int, optional): Number of input channels. If not provided, it will be inferred from the input data.
- **`strides`** (int or list of int): Stride size for the transposed convolution. Default is `[1, 1, 1]`.
- **`padding`** (str): Padding type, either `'VALID'` or `'SAME'`. Default is `'VALID'`.
- **`output_padding`** (int or list of int, optional): Additional size added to each dimension of the output shape. Default is `None`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format, either `'NDHWC'` or `'NCDHW'`. Default is `'NDHWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for dilated transposed convolution. Default is `None`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the 3D transposed convolution to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 3D transposed convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the conv3d_transpose layer
conv_layer = nn.conv3d_transpose(filters=32, kernel_size=3, input_size=64, strides=2, padding='SAME', activation='relu')

# Generate some sample data
data = tf.random.normal((10, 16, 16, 16, 64))  # Batch of 10 volumetric data, 16x16x16 voxels, 64 channels

# Apply the transposed convolutional layer
output = conv_layer(data)

print(output.shape)  # Output shape will depend on strides and padding
```

# cropping1d

This class implements 1D cropping for tensors.

**Initialization Parameters**

- **`cropping`** (int or list): The amount to crop from the start and end of the dimension. Can be an int or a list of two ints.

**Methods**

- **`__call__(self, data)`**: Applies 1D cropping to the input tensor.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 1D cropping.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the cropping1d layer
layer = nn.cropping1d(cropping=2)

# Define a sample input tensor
input_tensor = tf.random.normal(shape=(32, 100, 64))  # (batch_size, length, channels)

# Compute the cropped output
output = layer(input_tensor)

print(output.shape)  # Should be (32, 96, 64)
```

# cropping2d

This class implements 2D cropping for tensors.

**Initialization Parameters**

- **`cropping`** (int or list): The amount to crop from the dimensions. Can be an int, a list of two ints, or a list of four ints.

**Methods**

**`__call__(self, data)`**: Applies 2D cropping to the input tensor.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 2D cropping.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the cropping2d layer
layer = nn.cropping2d(cropping=[2, 3])

# Define a sample input tensor
input_tensor = tf.random.normal(shape=(32, 100, 100, 64))  # (batch_size, height, width, channels)

# Compute the cropped output
output = layer(input_tensor)

print(output.shape)  # Should be (32, 96, 94, 64)
```

# cropping3d

This class implements 3D cropping for tensors.

**Initialization Parameters**

- **`cropping`** (int or list): The amount to crop from the dimensions. Can be an int, a list of three ints, or a list of six ints.

**Methods**

**`__call__(self, data)`**: Applies 3D cropping to the input tensor.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the 3D cropping.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the cropping3d layer
layer = nn.cropping3d(cropping=[2, 3, 4])

# Define a sample input tensor
input_tensor = tf.random.normal(shape=(32, 50, 50, 50, 64))  # (batch_size, depth, height, width, channels)

# Compute the cropped output
output = layer(input_tensor)

print(output.shape)  # Should be (32, 46, 44, 42, 64)
```

# dense

The `dense` class implements a fully connected layer, which is a core component of many neural networks. This layer is used to perform a linear transformation on the input data, optionally followed by an activation function.

**Initialization Parameters**

- **`output_size`** (int): Number of output units (neurons) in the dense layer.
- **`input_size`** (int, optional): Number of input units (neurons) in the dense layer. If not provided, it will be inferred from the input data.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight matrix. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.
- **`name`** (str, optional): Name for the layer. Default is `None`.

**Methods**

- **`__call__(self, data)`**: Applies the dense layer to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the linear transformation and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the dense layer
dense_layer = nn.dense(output_size=64, input_size=128, activation='relu')

# Generate some sample data
data = tf.random.normal((10, 128))  # Batch of 10 samples, each with 128 features

# Apply the dense layer
output = dense_layer(data)

print(output.shape)  # Output shape will be (10, 64)
```

# depthwise_conv1d

The `depthwise_conv1d` class implements a depthwise 1D convolutional layer, which applies a single convolutional filter per input channel (channel-wise convolution), followed by an optional activation function.

**Initialization Parameters**

- **`kernel_size`** (int): Size of the convolutional kernel.
- **`depth_multiplier`** (int): Multiplier for the depth of the output tensor. Default is `1`.
- **`input_size`** (int, optional): Number of input channels. If not provided, it will be inferred from the input data.
- **`strides`** (int or list of int): Stride of the convolution. Default is `1`.
- **`padding`** (str): Padding algorithm to use. Default is `'VALID'`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format of the input and output data. Default is `'NHWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for the convolution. Default is `None`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the depthwise 1D convolutional layer to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the depthwise convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the depthwise_conv1d layer
depthwise_layer = nn.depthwise_conv1d(kernel_size=3, input_size=128, depth_multiplier=2, activation='relu')

# Generate some sample data
data = tf.random.normal((10, 128))  # Batch of 10 samples, each with 128 features

# Apply the depthwise 1D convolutional layer
output = depthwise_layer(data)

print(output.shape)  # Output shape will depend on the stride, padding, and depth_multiplier
```

# depthwise_conv2d

The `depthwise_conv2d` class implements a depthwise 2D convolutional layer, which applies a single convolutional filter per input channel (channel-wise convolution), followed by an optional activation function.

**Initialization Parameters**

- **`kernel_size`** (int or list of int): Size of the convolutional kernel. If an integer is provided, the same value will be used for both height and width.
- **`depth_multiplier`** (int): Multiplier for the depth of the output tensor. Default is `1`.
- **`input_size`** (int, optional): Number of input channels. If not provided, it will be inferred from the input data.
- **`strides`** (int or list of int): Stride of the convolution. Default is `1`.
- **`padding`** (str or list of int): Padding algorithm to use. Can be a string (`'VALID'` or `'SAME'`) or a list of integers for custom padding. Default is `'VALID'`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to use. Default is `None`.
- **`data_format`** (str): Data format of the input and output data. Default is `'NHWC'`.
- **`dilations`** (int or list of int, optional): Dilation rate for the convolution. Default is `None`.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the depthwise 2D convolutional layer to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the depthwise convolution and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the depthwise_conv2d layer
depthwise_layer = nn.depthwise_conv2d(kernel_size=3, input_size=128, depth_multiplier=2, activation='relu')

# Generate some sample data
data = tf.random.normal((10, 64, 64, 128))  # Batch of 10 samples, each with 64x64 size and 128 channels

# Apply the depthwise 2D convolutional layer
output = depthwise_layer(data)

print(output.shape)  # Output shape will depend on the stride, padding, and depth_multiplier
```

# dropout

The `dropout` class applies dropout to the input data, randomly dropping elements with a specified probability during training.

**Initialization Parameters**

- **`rate`** (float): The fraction of input units to drop, between 0 and 1.
- **`noise_shape`** (tensor, optional): A 1-D tensor representing the shape of the binary dropout mask that will be multiplied with the input. If `None`, the mask will have the same shape as the input.
- **`seed`** (int, optional): A random seed to ensure reproducibility.

**Methods**

- **`__call__(self, data, train_flag=None)`**: Applies dropout to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.
    - **`train_flag`** (bool, optional): If `True`, dropout is applied; if `False`, the input is returned unchanged. If `None`, the layer uses its internal `train_flag` attribute. Default is `None`.

  - **Returns**: The tensor after applying dropout during training or the original tensor during inference.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the dropout layer
dropout_layer = nn.dropout(rate=0.5)

# Generate some sample data
data = tf.random.normal((10, 64))  # Batch of 10 samples, each with 64 features

# Apply the dropout layer
output = dropout_layer(data, train_flag=True)

print(output.shape)  # Output shape will be the same as the input shape
```

# einsum_dense

This class implements a dense layer using `tf.einsum` for computations. It allows for flexible einsum operations on tensors of arbitrary dimensionality.

**Initialization Parameters**

- **`equation`** (str): An einsum equation string, e.g., `ab,bc->ac`.
- **`output_shape`** (int or list): The expected shape of the output tensor.
- **`input_shape`** (list, optional): Shape of the input tensor.
- **`activation`** (str, optional): Activation function to use.
- **`bias_axes`** (str, optional): Axes to apply bias on.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight matrix. Default is "Xavier".
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is "zeros".
- **`trainable`** (bool): Whether the layer's variables should be trainable. Default is `True`.
- **`dtype`** (str): Data type for the computations. Default is `'float32'`.

**Methods**

**`__call__(self, data)`**: Applies the einsum operation to the provided input tensor.

  - **Parameters**:
    - **`data`** (tensor): Input tensor.

  - **Returns**: Output tensor after applying the einsum operation and activation function (if specified).

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the einsum_dense layer
layer = nn.einsum_dense(equation='ab,bc->ac', output_shape=64, activation='relu')

# Define a sample input tensor
input_tensor = tf.random.normal(shape=(32, 128))  # (batch_size, input_dim)

# Compute the layer output
output = layer(input_tensor)

print(output.shape)  # Should be (32, 64)
```

# embedding

This class implements an embedding layer, which transforms input indices into dense vectors.

**Initialization Parameters**

- **`output_size`** (int): The size of the output embedding vectors.
- **`input_size`** (int, optional): The size of the input vocabulary. Default is `None`.
- **`initializer`** (str, list, tuple): The initializer for the embedding weights. Default is `'normal'`.
- **`sparse`** (bool): If `True`, supports sparse input tensors. Default is `False`.
- **`use_one_hot_matmul`** (bool): If `True`, uses one-hot matrix multiplication. Default is `False`.
- **`trainable`** (bool): If `True`, the embedding weights are trainable. Default is `True`.
- **`dtype`** (str): The data type for the embedding weights. Default is `'float32'`.

**Methods**

**`__call__(self, data)`**: Applies the embedding layer to the input indices.

  - **Parameters**:
    - **`data`** (tensor): The input tensor containing indices to be embedded.

  - **Returns**: The embedded output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the embedding class
embedding_layer = nn.embedding(output_size=64, input_size=1000)

# Define sample input data
input_data = tf.constant([1, 2, 3, 4, 5])

# Compute embeddings
output = embedding_layer(input_data)

print(output.shape)  # Should be (5, 64)
```

# FAVOR_attention

This class implements the Fast Attention via Positive Orthogonal Random Features (FAVOR) mechanism.

**Initialization Parameters**

- **`key_dim`** (int): The dimensionality of the keys.
- **`orthonormal`** (bool): If `True`, uses orthonormal random features. Default is `True`.
- **`causal`** (bool): If `True`, applies causal attention. Default is `False`.
- **`m`** (int): The number of random features. Default is `128`.
- **`redraw`** (bool): If `True`, redraws the random features at each call. Default is `True`.
- **`h`** (function, optional): A scaling function for the random features. Default is `None`.
- **`f`** (list): A list of activation functions to apply to the random features. Default is `[tf.nn.relu]`.
- **`randomizer`** (function): The function to generate random features. Default is `tf.random.normal`.
- **`eps`** (float): A small constant for numerical stability. Default is `0.0`.
- **`kernel_eps`** (float): A small constant added to the kernel features. Default is `0.001`.
- **`dtype`** (str): The data type for computations. Default is `'float32'`.

**Methods**

**`__call__(self, keys, values, queries)`**: Applies the FAVOR attention mechanism to the provided keys, values, and queries.

  - **Parameters**:
    - **`keys`** (Tensor): The key tensor.
    - **`values`** (Tensor): The value tensor.
    - **`queries`** (Tensor): The query tensor.

  - **Returns**:
    - **`Tensor`**: The result of the FAVOR attention mechanism.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the FAVOR attention class
attention_layer = nn.FAVOR_attention(key_dim=64)

# Define sample keys, values, and queries
keys = tf.random.normal(shape=(2, 10, 64))
values = tf.random.normal(shape=(2, 10, 64))
queries = tf.random.normal(shape=(2, 5, 64))

# Compute attention output
output = attention_layer(keys, values, queries)

print(output.shape)  # Should be (2, 5, 64)
```

# feed_forward_experts

This class implements a feed-forward layer with multiple experts, allowing for independent feed-forward blocks.

**Initialization Parameters**

- **`num_experts`** (int): The number of experts.
- **`d_ff`** (int): The dimension of the feed-forward layer of each expert.
- **`input_shape`** (tuple, optional): The shape of the input tensor. Default is `None`.
- **`inner_dropout`** (float): Dropout probability after intermediate activations. Default is `0.0`.
- **`output_dropout`** (float): Dropout probability after the output layer. Default is `0.0`.
- **`activation`** (function): The activation function. Default is `tf.nn.gelu`.
- **`kernel_initializer`** (str, list, tuple): The initializer for the kernel weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): The initializer for the bias weights. Default is `'zeros'`.

**Methods**

**`__call__(self, data, train_flag=True)`**: Applies the feed-forward experts layer to the input data.

  - **Parameters**:
    - **`data`** (Tensor): The input tensor.
    - **`train_flag`** (bool): If `True`, applies dropout during training.

  - **Returns**: The transformed input tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the feed-forward experts class
ff_experts_layer = nn.feed_forward_experts(num_experts=4, d_ff=128, input_shape=(None, 4, 32, 64))

# Define sample input data
input_data = tf.random.normal(shape=(8, 4, 32, 64))

# Compute output
output = ff_experts_layer(input_data, train_flag=True)

print(output.shape)  # Should be (8, 4, 32, 64)
```

# filter_response_norm

This class implements the Filter Response Normalization (FRN) layer, which normalizes per-channel activations.

**Initialization Parameters**

- **`input_shape`** (tuple, optional): The shape of the input tensor. Default is `None`.
- **`epsilon`** (float): Small constant added to variance to avoid division by zero. Default is `1e-6`.
- **`axis`** (list): List of axes that should be normalized. Default is `[1, 2]`.
- **`beta_initializer`** (str, list, tuple): Initializer for the beta weights. Default is `'zeros'`.
- **`gamma_initializer`** (str, list, tuple): Initializer for the gamma weights. Default is `'ones'`.
- **`learned_epsilon`** (bool): If `True`, adds a learnable epsilon parameter. Default is `False`.
- **`dtype`** (str): The data type for computations. Default is `'float32'`.

**Methods**

**`__call__(self, data)`**: Applies the FRN layer to the input data.

  - **Parameters**:
    - **`data`** (Tensor): The input tensor.

- **Returns**: The normalized output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the FRN class
frn_layer = nn.filter_response_norm(input_shape=(None, 32, 32, 64))

# Define sample input data
input_data = tf.random.normal(shape=(8, 32, 32, 64))

# Compute output
output = frn_layer(input_data)

print(output.shape)  # Should be (8, 32, 32, 64)
```

# flatten

This class implements a flatten layer, which reshapes the input tensor to a 2D tensor.

**Methods**

**`__call__(self, data)`**: Applies the flatten layer to the input data.

  - **Parameters**:
    - **`data`** (Tensor): The input tensor.

- **Returns**: The reshaped output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Instantiate the flatten class
flatten_layer = nn.flatten()

# Define sample input data
input_data = tf.random.normal(shape=(8, 32, 32, 64))

# Compute output
output = flatten_layer(input_data)

print(output.shape)  # Should be (8, 65536)
```

# gaussian_dropout

This class applies multiplicative 1-centered Gaussian noise, useful for regularization during training.

**Initialization Parameters**

- **`rate`** (float): Drop probability. The noise will have standard deviation `sqrt(rate / (1 - rate))`.
- **`seed`** (int, optional): Random seed for deterministic behavior. Default is `7`.

**Methods**

**`__call__(self, data, train_flag=True)`**: Applies the Gaussian Dropout to the input tensor during training.

  - **Parameters**:
    - **`data`** (Tensor): Input tensor of any rank.
    - **`train_flag`** (bool): If `True`, applies dropout. If `False`, returns the input tensor as is.

  - Returns: The output tensor with Gaussian Dropout applied during training.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

dropout_layer = nn.gaussian_dropout(rate=0.5, seed=7)
data = tf.random.normal(shape=(3, 4))
output = dropout_layer(data, train_flag=True)

print(output.shape)  # Same shape as input
```

# gaussian_noise

This class applies additive zero-centered Gaussian noise, useful for regularization and data augmentation during training.

**Initialization Parameters**

- **`stddev`** (float): Standard deviation of the noise distribution.
- **`seed`** (int, optional): Random seed for deterministic behavior. Default is `7`.

**Methods**

**`__call__(self, data, train_flag=True)`**: Applies Gaussian noise to the input tensor during training.

  - **Parameters**:
    - **`data`** (Tensor): Input tensor of any rank.
    - **`train_flag`** (bool): If `True`, adds noise. If `False`, returns the input tensor as is.

  - **Returns**: The output tensor with Gaussian noise added during training.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

noise_layer = nn.gaussian_noise(stddev=0.1, seed=7)
data = tf.random.normal(shape=(3, 4))
output = noise_layer(data, train_flag=True)

print(output.shape)  # Same shape as input
```

# GCN

This class implements a multi-layer Graph Convolutional Network (GCN).

**Initialization Parameters**

- **`x_dim`** (int): Dimension of input features.
- **`h_dim`** (int): Dimension of hidden layers.
- **`out_dim`** (int): Dimension of output features.
- **`nb_layers`** (int): Number of GCN layers. Default is `2`.
- **`dropout_rate`** (float): Dropout rate for regularization. Default is `0.5`.
- **`bias`** (bool): If `True`, adds a learnable bias to the output. Default is `True`.

**Methods**

**`__call__(self, x, adj)`**: Applies the multi-layer GCN to the input tensor and adjacency matrix.

  - **Parameters**:
    - **`x`** (Tensor): Input feature tensor.
    - **`adj`** (Tensor): Adjacency matrix tensor.

  - **Returns**: The output tensor after applying the GCN layers.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

gcn = nn.GCN(x_dim=10, h_dim=20, out_dim=5, nb_layers=2, dropout_rate=0.5)
x = tf.random.normal(shape=(5, 10))
adj = tf.eye(5)

output = gcn(x, adj)
print(output.shape)  # Should be (5, 5)
```

# global_avg_pool1d

These classes implement global average pooling operations for 1D tensors.

**Initialization Parameters**

- **`keepdims`** (bool): If `True`, retains reduced dimensions with length 1. Default is `False`.

**Methods**

**`__call__(self, data)`**: Applies global average pooling to the input tensor.

  - **Parameters**:
    - **`data`** (Tensor): Input 1D tensor.

  - **Returns**: The pooled tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

pool1d = nn.global_avg_pool1d(keepdims=False)
data = tf.random.normal(shape=(3, 4, 5))
output = pool1d(data)

print(output.shape)  # Should be (3, 5)
```

# global_avg_pool2d

These classes implement global average pooling operations for 2D tensors.

**Initialization Parameters**

- **`keepdims`** (bool): If `True`, retains reduced dimensions with length 1. Default is `False`.

**Methods**

**`__call__(self, data)`**: Applies global average pooling to the input tensor.

  - **Parameters**:
    - **`data`** (Tensor): Input 2D tensor.

  - **Returns**: The pooled tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

pool2d = nn.global_avg_pool2d(keepdims=False)
data = tf.random.normal(shape=(3, 4, 5, 6))
output = pool2d(data)

print(output.shape)  # Should be (3, 6)
```

# global_avg_pool3d

These classes implement global average pooling operations for 3D tensors.

**Initialization Parameters**

- **`keepdims`** (bool): If `True`, retains reduced dimensions with length 1. Default is `False`.

**Methods**

**`__call__(self, data)`**: Applies global average pooling to the input tensor.

  - **Parameters**:
    - **`data`** (Tensor): Input 3D tensor.

  - **Returns**: The pooled tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

pool3d = nn.global_avg_pool3d(keepdims=False)
data = tf.random.normal(shape=(3, 4, 5, 6, 7))
output = pool3d(data)

print(output.shape)  # Should be (3, 7)
```

# group_norm

The `group_norm` class implements Group Normalization, a technique that divides channels into groups and normalizes each group independently. This can be more stable than batch normalization for small batch sizes.

**Initialization Parameters**

- **`groups`** (int, default=32): Number of groups for normalization. Must be a divisor of the number of channels.
- **`input_size`** (int, optional): Size of the input dimension. If not provided, it will be inferred from the input data.
- **`axis`** (int or list/tuple, default=-1): Axis or axes to normalize across. Typically the feature axis.
- **`epsilon`** (float, default=1e-3): Small constant to avoid division by zero.
- **`center`** (bool, default=True): If `True`, add offset `beta`.
- **`scale`** (bool, default=True): If `True`, multiply by `gamma`.
- **`beta_initializer`** (str, list, tuple): Initializer for the beta parameter. Default is `'zeros'`.
- **`gamma_initializer`** (str, list, tuple): Initializer for the gamma parameter. Default is `'ones'`.
- **`mask`** (tensor, optional): Mask tensor for weighted mean and variance calculation.
- **`dtype`** (str, default='float32'): Data type for the layer parameters.

**Methods**

- **`__call__(self, data)`**: Applies group normalization to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of any rank.

  - **Returns**: The normalized tensor with the same shape as the input.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the group normalization layer
group_norm_layer = nn.group_norm(groups=32, input_size=64)

# Generate some sample data
data = tf.random.normal((10, 64))  # Batch of 10 samples, each with 64 features

# Apply the group normalization layer
output = group_norm_layer(data)

print(output.shape)  # Output shape will be the same as the input shape
```

# GRU

The `GRU` class implements a Gated Recurrent Unit (GRU) layer, a type of recurrent neural network layer designed to handle sequential data. GRUs are known for their efficiency in learning and processing long sequences.

**Initialization Parameters**

- **`output_size`** (int): Size of the output dimension.
- **`input_size`** (int, optional): Size of the input dimension. If not provided, it will be inferred from the input data.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight matrices. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vectors. Default is `'zeros'`.
- **`return_sequence`** (bool, default=False): If `True`, returns the full sequence of outputs. If `False`, returns only the final output.
- **`use_bias`** (bool, default=True): If `True`, includes bias terms in the calculations.
- **`trainable`** (bool, default=True): If `True`, the layer's parameters will be trainable.
- **`dtype`** (str, default='float32'): Data type for the layer's parameters.

**Methods**

- **`__call__(self, data)`**: Applies the GRU layer to the input `data`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, timesteps, input_size]`.

  - **Returns**: 
    - If `return_sequence` is `True`, returns a tensor of shape `[batch_size, timesteps, output_size]`.
    - If `return_sequence` is `False`, returns a tensor of shape `[batch_size, output_size]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the GRU layer
gru_layer = nn.GRU(output_size=64, input_size=128, return_sequence=True)

# Generate some sample data
data = tf.random.normal((32, 10, 128))  # Batch of 32 sequences, each with 10 timesteps, each timestep with 128 features

# Apply the GRU layer
output = gru_layer(data)

print(output.shape)  # Output shape will be (32, 10, 64) if return_sequence is True
```

# GRUCell

The `GRUCell` class implements a single Gated Recurrent Unit (GRU) cell, a building block for recurrent neural networks designed to handle sequential data. GRU cells are efficient in learning and processing long sequences by controlling the flow of information with reset and update gates.

**Initialization Parameters**

- **`weight_shape`** (tuple): Shape of the weight matrix, typically `[input_size + output_size, output_size]`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight matrix. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`use_bias`** (bool, default=True): If `True`, includes bias terms in the calculations.
- **`trainable`** (bool, default=True): If `True`, the cell's parameters will be trainable.
- **`dtype`** (str, default='float32'): Data type for the cell's parameters.

**Methods**

- **`__call__(self, data, state)`**: Applies the GRU cell to the input `data` and previous `state`.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, input_size]`.
    - **`state`** (tensor): Previous hidden state tensor of shape `[batch_size, output_size]`.

  - **Returns**: 
    - **`output`** (tensor): Output tensor of shape `[batch_size, output_size]`, same as the new hidden state.
    - **`h_new`** (tensor): New hidden state tensor of shape `[batch_size, output_size]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the GRUCell
gru_cell = nn.GRUCell(weight_shape=(128, 64))

# Generate some sample data
data = tf.random.normal((32, 128))  # Batch of 32 samples, each with 128 features
state = tf.zeros((32, 64))          # Initial state with 32 samples, each with 64 features

# Apply the GRU cell
output, new_state = gru_cell(data, state)

print(output.shape)  # Output shape will be (32, 64)
print(new_state.shape)  # New state shape will be (32, 64)
```

# identity

The `identity` class implements an identity layer, which is a simple layer that outputs the input data unchanged. This layer can be useful in various neural network architectures for maintaining the shape of data or as a placeholder.

**Initialization Parameters**

- **`input_size`** (int, optional): Size of the input. If provided, it will set the `output_size` to be the same as the `input_size`.

**Methods**

- **`__call__(self, data)`**: Returns the input data unchanged.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of any shape.
    
  - **Returns**: 
    - **`data`** (tensor): Output tensor, which is identical to the input tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the identity layer
identity_layer = nn.identity(input_size=128)

# Generate some sample data
data = tf.random.normal((32, 128))  # Batch of 32 samples, each with 128 features

# Apply the identity layer
output = identity_layer(data)

print(output.shape)  # Output shape will be (32, 128), same as input shape
print(tf.reduce_all(tf.equal(data, output)))  # Should print True, indicating the output is the same as input
```

# LSTM

The `LSTM` class implements a long short-term memory (LSTM) layer, which is a type of recurrent neural network (RNN) used for processing sequential data. LSTMs are designed to capture long-term dependencies and are commonly used in tasks such as time series forecasting, natural language processing, and more.

**Initialization Parameters**

- **`output_size`** (int): The size of the output vector for each time step.
- **`input_size`** (int, optional): The size of the input vector. If not provided, it will be inferred from the input data.
- **`weight_initializer`** (str, list, tuple): The method to initialize weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): The method to initialize biases. Default is `'zeros'`.
- **`return_sequence`** (bool, default=False): Whether to return the full sequence of outputs or just the last output.
- **`use_bias`** (bool, default=True): Whether to use bias vectors.
- **`trainable`** (bool, default=True): Whether the layer parameters are trainable.
- **`dtype`** (str, default='float32'): The data type of the layer parameters.

**Methods**

- **`__call__(self, data)`**: Processes the input data through the LSTM layer and returns the output.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, time_steps, input_size]`.
    
  - **Returns**: 
    - **`output`** (tensor): Output tensor. If `return_sequence` is `True`, the shape is `[batch_size, time_steps, output_size]`. Otherwise, the shape is `[batch_size, output_size]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the LSTM layer
lstm_layer = nn.LSTM(output_size=128, input_size=64, return_sequence=True)

# Generate some sample data
data = tf.random.normal((32, 10, 64))  # Batch of 32 samples, each with 10 time steps and 64 features

# Apply the LSTM layer
output = lstm_layer(data)

print(output.shape)  # Output shape will be (32, 10, 128) if return_sequence is True
```

# LSTMCell

The `LSTMCell` class implements a long short-term memory (LSTM) cell, a fundamental building block for LSTM layers in recurrent neural networks (RNNs). LSTM cells are designed to capture long-term dependencies in sequential data, making them suitable for tasks such as time series forecasting, natural language processing, and more.

**Initialization Parameters**

- **`weight_shape`** (tuple): Shape of the weights. Should be `[input_size, hidden_size]`.
- **`weight_initializer`** (str, list, tuple): Method for weight initialization. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Method for bias initialization. Default is `'zeros'`.
- **`use_bias`** (bool, default=True): Whether to use bias vectors.
- **`trainable`** (bool, default=True): Whether the layer parameters are trainable.
- **`dtype`** (str, default='float32'): Data type of the layer parameters.

**Methods**

- **`__call__(self, data, state)`**: Processes the input data through the LSTM cell and returns the output and new cell state.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, input_size]`.
    - **`state`** (tensor): Previous hidden state tensor of shape `[batch_size, hidden_size]`.
    
  - **Returns**: 
    - **`output`** (tensor): Output tensor of shape `[batch_size, hidden_size]`.
    - **`c_new`** (tensor): New cell state tensor of shape `[batch_size, hidden_size]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Define input size and hidden size
input_size = 64
hidden_size = 128

# Create an instance of the LSTMCell
lstm_cell = nn.LSTMCell(weight_shape=(input_size, hidden_size))

# Generate some sample data
data = tf.random.normal((32, input_size))  # Batch of 32 samples, each with 64 features
state = tf.random.normal((32, hidden_size))  # Batch of 32 samples, each with 128 features for the previous state

# Apply the LSTM cell
output, new_state = lstm_cell(data, state)

print(output.shape)  # Output shape will be (32, 128)
print(new_state.shape)  # New state shape will be (32, 128)
```

# layer_norm

The `layer_norm` class implements layer normalization, a technique used to normalize the inputs across the features of a layer. This normalization helps stabilize and accelerate the training of deep neural networks.

**Initialization Parameters**

- **`input_size`** (int, default=None): Size of the input features.
- **`axis`** (int or list of ints, default=-1): Axis or axes along which to normalize.
- **`momentum`** (float, default=0.99): Momentum for the moving average.
- **`epsilon`** (float, default=0.001): Small value to avoid division by zero.
- **`center`** (bool, default=True): Whether to include a beta parameter.
- **`scale`** (bool, default=True): Whether to include a gamma parameter.
- **`rms_scaling`** (bool, default=False): Whether to use RMS scaling.
- **`beta_initializer`** (str, list, tuple): Initializer for the beta parameter. Default is `'zeros'`.
- **`gamma_initializer`** (str, list, tuple): Initializer for the gamma parameter. Default is `'ones'`.
- **`dtype`** (str, default='float32'): Data type of the layer parameters.

**Methods**

- **`__call__(self, data)`**: Applies layer normalization to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, ..., input_size]`.

  - **Returns**: 
    - **`outputs`** (tensor): Normalized tensor of the same shape as input.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of layer_norm
layer_norm_layer = nn.layer_norm(input_size=128)

# Generate some sample data
data = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 timesteps, 128 features

# Apply layer normalization
normalized_data = layer_norm_layer(data)

print(normalized_data.shape)  # Output shape will be (32, 10, 128)
```

# multihead_attention

The `multihead_attention` class implements multi-head attention, a core component in transformer models that allows the model to focus on different parts of the input sequence when generating each part of the output sequence.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`input_size`** (int, default=None): Size of the input features.
- **`kdim`** (int, default=None): Dimension of the key vectors (defaults to `input_size` if not specified).
- **`vdim`** (int, default=None): Dimension of the value vectors (defaults to `input_size` if not specified).
- **`weight_initializer`** (str, list, tuple): Initializer for the weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the biases. Default is `'zeros'`.
- **`use_bias`** (bool, default=True): Whether to use biases in the dense layers.
- **`dtype`** (str, default='float32'): Data type of the layer parameters.

**Methods**

- **`__call__(self, target, source=None, mask=None)`**: Applies multi-head attention to the input data.

  - **Parameters**:
    - **`target`** (tensor): Target sequence tensor of shape `[batch_size, seq_length, input_size]`.
    - **`source`** (tensor, default=None): Source sequence tensor of shape `[batch_size, seq_length, input_size]`. If `None`, self-attention is applied.
    - **`mask`** (tensor, default=None): Mask tensor to apply during attention calculation.

  - **Returns**: 
    - **`output`** (tensor): Output tensor of shape `[batch_size, seq_length, input_size]`.
    - **`attention_weights`** (tensor): Attention weights tensor of shape `[batch_size, n_head, seq_length, seq_length]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of multihead_attention
mha_layer = nn.multihead_attention(n_head=8, input_size=128)

# Generate some sample data
target = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 timesteps, 128 features

# Apply multihead attention
output, attention_weights = mha_layer(target)

print(output.shape)  # Output shape will be (32, 10, 128)
print(attention_weights.shape)  # Attention weights shape will be (32, 8, 10, 10)
```

# RNN

The `RNN` class implements a simple Recurrent Neural Network (RNN) layer, which processes sequential data one timestep at a time and maintains a state that is updated at each timestep.

**Initialization Parameters**

- **`output_size`** (int): Size of the output features.
- **`input_size`** (int, default=None): Size of the input features. If not specified, it will be inferred from the input data.
- **`weight_initializer`** (str, list, tuple): Initializer for the weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the biases. Default is `'zeros'`.
- **`activation`** (str, default=None): Activation function to use (should be a key in `activation_dict`).
- **`return_sequence`** (bool, default=False): Whether to return the full sequence of outputs or just the last output.
- **`use_bias`** (bool, default=True): Whether to use biases in the RNN cell.
- **`trainable`** (bool, default=True): Whether the layer parameters are trainable.
- **`dtype`** (str, default='float32'): Data type of the layer parameters.

**Methods**

- **`__call__(self, data)`**: Applies the RNN layer to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input data tensor of shape `[batch_size, timestep, input_size]`.

  - **Returns**:
    - **`output`** (tensor): Output tensor of shape `[batch_size, timestep, output_size]` if `return_sequence=True`, otherwise `[batch_size, output_size]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of RNN
rnn_layer = nn.RNN(output_size=64, input_size=128, activation='tanh')

# Generate some sample data
data = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 timesteps, 128 features

# Apply the RNN layer
output = rnn_layer(data)

print(output.shape)  # Output shape will be (32, 10, 64) if return_sequence=True, otherwise (32, 64)
```

# RNNCell

The `RNNCell` class implements a basic Recurrent Neural Network (RNN) cell, which processes a single timestep of input data and maintains a state that is updated at each step.

**Initialization Parameters**

- **`weight_shape`** (tuple): Shape of the weight matrix for input data.
- **`weight_initializer`** (str, list, tuple): Initializer for the weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the biases. Default is `'zeros'`.
- **`activation`** (str, default=None): Activation function to use (should be a key in `activation_dict`).
- **`use_bias`** (bool, default=True): Whether to use biases in the RNN cell.
- **`trainable`** (bool, default=True): Whether the layer parameters are trainable.
- **`dtype`** (str, default='float32'): Data type of the layer parameters.

**Methods**

- **`__call__(self, data, state)`**: Applies the RNN cell to the input data and updates the state.

  - **Parameters**:
    - **`data`** (tensor): Input data tensor of shape `[batch_size, input_size]`.
    - **`state`** (tensor): Previous state tensor of shape `[batch_size, output_size]`.

  - **Returns**:
    - **`output`** (tensor): Output tensor of shape `[batch_size, output_size]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of RNNCell
rnn_cell = nn.RNNCell(weight_shape=(128, 64), activation='tanh')

# Generate some sample data
data = tf.random.normal((32, 128))  # Batch of 32 samples, 128 features
state = tf.zeros((32, 64))  # Initial state of zeros

# Apply the RNN cell
output = rnn_cell(data, state)

print(output.shape)  # Output shape will be (32, 64)
```

# separable_conv1d

The `separable_conv1d` class implements a 1D separable convolutional layer, which is a depthwise separable convolution that reduces the number of parameters and computation cost compared to a regular convolutional layer. This layer is often used in neural network models for processing sequential data, such as time-series or audio signals.

**Initialization Parameters**

- **`filters`** (int): The number of output filters in the convolution.
- **`kernel_size`** (int): The length of the convolution window.
- **`depth_multiplier`** (int): The number of depthwise convolution output channels for each input channel.
- **`input_size`** (int, default=None): The number of input channels. If not specified, it will be inferred from the input data.
- **`strides`** (list, default=[1]): The stride length of the convolution.
- **`padding`** (str, default='VALID'): One of `"VALID"` or `"SAME"`.
- **`data_format`** (str, default='NHWC'): The data format, either `"NHWC"` or `"NCHW"`.
- **`dilations`** (int or list, default=None): The dilation rate to use for dilated convolution.
- **`weight_initializer`** (str, list, tuple): Initializer for the weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the biases. Default is `'zeros'`.
- **`activation`** (str, default=None): Activation function to apply.
- **`use_bias`** (bool, default=True): Whether to use a bias vector.
- **`trainable`** (bool, default=True): Whether the layer's variables should be trainable.
- **`dtype`** (str, default='float32'): Data type of the layer parameters.

**Methods**

- **`__call__(self, data)`**: Applies the separable convolution to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, seq_length, input_size]`.

  - **Returns**:
    - **`output`** (tensor): Output tensor of shape `[batch_size, new_seq_length, filters]`, where `new_seq_length` depends on the padding and strides.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of separable_conv1d
sep_conv_layer = nn.separable_conv1d(filters=64, kernel_size=3, depth_multiplier=2, input_size=128)

# Generate some sample data
data = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 timesteps, 128 features

# Apply the separable convolution
output = sep_conv_layer(data)

print(output.shape)  # Output shape will depend on the padding and strides
```

# stochastic_depth

The `stochastic_depth` class implements a layer that randomly drops entire paths (blocks of layers) during training, helping to regularize the model and prevent overfitting. This technique is also known as DropPath.

**Initialization Parameters**

- **`drop_path_rate`** (float): The probability of dropping a path. Must be between 0 and 1.

**Methods**

- **`__call__(self, x, train_flag=None)`**: Applies stochastic depth to the input tensor during training.

  - **Parameters**:
    - **`x`** (tensor): Input tensor.
    - **`train_flag`** (bool, default=None): Whether to apply stochastic depth (drop paths). If `None`, uses the internal training flag.

  - **Returns**:
    - **`output`** (tensor): Output tensor with some paths randomly dropped during training.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of stochastic_depth
drop_layer = nn.stochastic_depth(drop_path_rate=0.2)

# Generate some sample data
data = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 timesteps, 128 features

# Apply stochastic depth during training
output = drop_layer(data, train_flag=True)

print(output.shape)  # Output shape will be the same as input shape
```

# Transformer

The `Transformer` class implements the Transformer model, which is widely used in natural language processing tasks due to its efficiency and scalability in handling sequential data. This class combines both the encoder and decoder components of the Transformer architecture.

**Initialization Parameters**

- **`d_model`** (int, default=512): The number of expected features in the input.
- **`nhead`** (int, default=8): The number of heads in the multihead attention mechanism.
- **`num_encoder_layers`** (int, default=6): The number of encoder layers in the Transformer.
- **`num_decoder_layers`** (int, default=6): The number of decoder layers in the Transformer.
- **`dim_feedforward`** (int, default=2048): The dimension of the feedforward network model.
- **`dropout`** (float, default=0.1): The dropout value.
- **`activation`** (function, default=tf.nn.relu): The activation function of the intermediate layer.
- **`custom_encoder`** (optional): Custom encoder instance to override the default encoder.
- **`custom_decoder`** (optional): Custom decoder instance to override the default decoder.
- **`layer_norm_eps`** (float, default=1e-5): The epsilon value for layer normalization.
- **`norm_first`** (bool, default=False): Whether to apply normalization before or after each sub-layer.
- **`bias`** (bool, default=True): Whether to use bias in the layers.
- **`dtype`** (str, default='float32'): Data type of the layer parameters.

**Methods**

- **`__call__(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, train_flag=True)`**: Applies the Transformer model to the input data.

  - **Parameters**:
    - **`src`** (tensor): Source sequence tensor of shape `[batch_size, src_seq_length, d_model]`.
    - **`tgt`** (tensor): Target sequence tensor of shape `[batch_size, tgt_seq_length, d_model]`.
    - **`src_mask`** (tensor, default=None): Optional mask for the source sequence.
    - **`tgt_mask`** (tensor, default=None): Optional mask for the target sequence.
    - **`memory_mask`** (tensor, default=None): Optional mask for the memory sequence.
    - **`train_flag`** (bool, default=True): Flag indicating whether the model is in training mode.

  - **Returns**:
    - **`output`** (tensor): Output tensor of shape `[batch_size, tgt_seq_length, d_model]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of Transformer
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

# Generate some sample data
src = tf.random.normal((32, 10, 512))  # Batch of 32 samples, 10 source timesteps, 512 features
tgt = tf.random.normal((32, 10, 512))  # Batch of 32 samples, 10 target timesteps, 512 features

# Apply the Transformer model
output = transformer(src, tgt)

print(output.shape)  # Output shape will be (32, 10, 512)
```

# zeropadding1d

The `zeropadding1d` class implements a 1D zero-padding layer, which pads the input tensor along the second dimension (time steps) with zeros.

**Initialization Parameters**

- **`input_size`** (int, default=None): Size of the input features. If provided, `output_size` is set to `input_size`.
- **`padding`** (int or tuple of 2 ints, default=None): Amount of padding to add to the beginning and end of the time steps. If not specified, padding can be set during the `__call__` method.

**Methods**

- **`__call__(self, data, padding=1)`**: Applies zero-padding to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, time_steps, features]`.
    - **`padding`** (int or tuple of 2 ints, default=1): Amount of padding to add if not specified during initialization.

  - **Returns**:
    - **`output`** (tensor): Output tensor with zero-padding applied.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of zeropadding1d
padding_layer = nn.zeropadding1d(padding=(1, 2))

# Generate some sample data
data = tf.random.normal((32, 10, 128))  # Batch of 32 samples, 10 timesteps, 128 features

# Apply zero-padding
output = padding_layer(data)

print(output.shape)  # Output shape will be (32, 13, 128)
```

# zeropadding2d

The `zeropadding2d` class implements a 2D zero-padding layer, which pads the input tensor along the height and width dimensions with zeros.

**Initialization Parameters**

- **`input_size`** (int, default=None): Size of the input features. If provided, `output_size` is set to `input_size`.
- **`padding`** (int or tuple of 2 tuples, default=None): Amount of padding to add to the height and width dimensions. Each tuple contains two integers representing the padding before and after each dimension. If not specified, padding can be set during the `__call__` method.

**Methods**

- **`__call__(self, data, padding=(1, 1))`**: Applies zero-padding to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, height, width, channels]`.
    - **`padding`** (int or tuple of 2 tuples, default=(1, 1)): Amount of padding to add if not specified during initialization.

  - **Returns**:
    - **`output`** (tensor): Output tensor with zero-padding applied.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of zeropadding2d
padding_layer = nn.zeropadding2d(padding=((1, 2), (3, 4)))

# Generate some sample data
data = tf.random.normal((32, 28, 28, 3))  # Batch of 32 samples, 28x28 image size, 3 channels

# Apply zero-padding
output = padding_layer(data)

print(output.shape)  # Output shape will be (32, 31, 35, 3)
```

# zeropadding3d

The `zeropadding3d` class implements a 3D zero-padding layer, which pads the input tensor along three spatial dimensions with zeros.

**Initialization Parameters**

- **`input_size`** (int, default=None): Size of the input features. If provided, `output_size` is set to `input_size`.
- **`padding`** (int or tuple of 3 ints or tuple of 3 tuples of 2 ints, default=(1, 1, 1)): Amount of padding to add to each of the three spatial dimensions. If not specified, padding can be set during the `__call__` method.

**Methods**

- **`__call__(self, data, padding=(1, 1, 1))`**: Applies zero-padding to the input data.

  - **Parameters**:
    - **`data`** (tensor): Input tensor of shape `[batch_size, depth, height, width, channels]`.
    - **`padding`** (int or tuple of 3 ints or tuple of 3 tuples of 2 ints, default=(1, 1, 1)): Amount of padding to add if not specified during initialization.

  - **Returns**:
    - **`output`** (tensor): Output tensor with zero-padding applied.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of zeropadding3d
padding_layer = nn.zeropadding3d(padding=(2, 3, 4))

# Generate some sample data
data = tf.random.normal((32, 10, 10, 10, 3))  # Batch of 32 samples, 10x10x10 volumes, 3 channels

# Apply zero-padding
output = padding_layer(data)

print(output.shape)  # Output shape will be (32, 14, 16, 18, 3)
```
