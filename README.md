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

# FastAdaptiveAvgPool

The `FastAdaptiveAvgPool` class implements fast adaptive average pooling for 2D inputs. It computes the average of the input tensor along the spatial dimensions.

**Initialization Parameters**

- **`flatten`** (bool, optional): If `True`, flattens the output. Default is `False`.
- **`input_fmt`** (str, optional): Specifies the format of the input tensor (`'NHWC'` or `'NCHW'`). Default is `'NHWC'`.

**Methods**

- **`__call__(self, x)`**: Applies adaptive average pooling to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of FastAdaptiveAvgPool
avg_pool = nn.FastAdaptiveAvgPool(flatten=True)

# Generate some sample data
data = tf.random.normal((2, 8, 8, 3))

# Apply average pooling
output = avg_pool(data)
```

# FastAdaptiveMaxPool

The `FastAdaptiveMaxPool` class implements fast adaptive max pooling for 2D inputs. It computes the maximum value of the input tensor along the spatial dimensions.

**Initialization Parameters**

- **`flatten`** (bool, optional): If `True`, flattens the output. Default is `False`.
- **`input_fmt`** (str, optional): Specifies the format of the input tensor (`'NHWC'` or `'NCHW'`). Default is `'NHWC'`.

**Methods**

- **`__call__(self, x)`**: Applies adaptive max pooling to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of FastAdaptiveMaxPool
max_pool = nn.FastAdaptiveMaxPool(flatten=True)

# Generate some sample data
data = tf.random.normal((2, 8, 8, 3))

# Apply max pooling
output = max_pool(data)
```

# FastAdaptiveAvgMaxPool

The `FastAdaptiveAvgMaxPool` class combines both average and max pooling for 2D inputs. It computes the average and maximum of the input tensor along the spatial dimensions and returns their mean.

**Initialization Parameters**

- **`flatten`** (bool, optional): If `True`, flattens the output. Default is `False`.
- **`input_fmt`** (str, optional): Specifies the format of the input tensor (`'NHWC'` or `'NCHW'`). Default is `'NHWC'`.

**Methods**

- **`__call__(self, x)`**: Applies combined adaptive average and max pooling to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of FastAdaptiveAvgMaxPool
avg_max_pool = nn.FastAdaptiveAvgMaxPool(flatten=True)

# Generate some sample data
data = tf.random.normal((2, 8, 8, 3))

# Apply average and max pooling
output = avg_max_pool(data)
```

# FastAdaptiveCatAvgMaxPool

The `FastAdaptiveCatAvgMaxPool` class concatenates the results of both average and max pooling for 2D inputs. It computes the average and maximum of the input tensor along the spatial dimensions and concatenates the results.

**Initialization Parameters**

- **`flatten`** (bool, optional): If `True`, flattens the output. Default is `False`.
- **`input_fmt`** (str, optional): Specifies the format of the input tensor (`'NHWC'` or `'NCHW'`). Default is `'NHWC'`.

**Methods**

- **`__call__(self, x)`**: Applies concatenated adaptive average and max pooling to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Concatenated pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of FastAdaptiveCatAvgMaxPool
cat_avg_max_pool = nn.FastAdaptiveCatAvgMaxPool(flatten=True)

# Generate some sample data
data = tf.random.normal((2, 8, 8, 3))

# Apply concatenated average and max pooling
output = cat_avg_max_pool(data)
```

# AdaptiveAvgMaxPool2d

The `AdaptiveAvgMaxPool2d` class implements adaptive pooling that combines average and max pooling for 2D inputs.

**Initialization Parameters**

- **`output_size`** (tuple of int, optional): Specifies the output size. Default is `1`.

**Methods**

- **`__call__(self, x)`**: Applies adaptive average and max pooling to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of AdaptiveAvgMaxPool2d
adaptive_avg_max_pool = nn.AdaptiveAvgMaxPool2d(output_size=(2, 2))

# Generate some sample data
data = tf.random.normal((2, 8, 8, 3))

# Apply adaptive average and max pooling
output = adaptive_avg_max_pool(data)
```

# AdaptiveCatAvgMaxPool2d

The `AdaptiveCatAvgMaxPool2d` class implements adaptive pooling that concatenates the results of average and max pooling for 2D inputs.

**Initialization Parameters**

- **`output_size`** (tuple of int, optional): Specifies the output size. Default is `1`.

**Methods**

- **`__call__(self, x)`**: Applies adaptive concatenated average and max pooling to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Concatenated pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of AdaptiveCatAvgMaxPool2d
adaptive_cat_avg_max_pool = nn.AdaptiveCatAvgMaxPool2d(output_size=(2, 2))

# Generate some sample data
data = tf.random.normal((2, 8, 8, 3))

# Apply adaptive concatenated average and max pooling
output = adaptive_cat_avg_max_pool(data)
```

# SelectAdaptivePool2d

The `SelectAdaptivePool2d` class provides a selectable global pooling layer with dynamic input kernel size.

**Initialization Parameters**

- **`output_size`** (tuple of int, optional): Specifies the output size. Default is `1`.
- **`pool_type`** (str, optional): Specifies the type of pooling (`'fast'`, `'avgmax'`, `'catavgmax'`, `'max'`, or `'avg'`). Default is `'fast'`.
- **`flatten`** (bool, optional): If `True`, flattens the output. Default is `False`.
- **`input_fmt`** (str, optional): Specifies the format of the input tensor (`'NHWC'` or `'NCHW'`). Default is `'NHWC'`.

**Methods**

- **`is_identity(self)`**: Checks if the pool type is an identity (no pooling).
- **`__call__(self, x)`**: Applies the selected pooling method to the input `x`.
- **`feat_mult(self)`**: Returns the feature multiplier for the selected pooling method.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Pooled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of SelectAdaptivePool2d
select_pool = nn.SelectAdaptivePool2d(pool_type='avgmax', flatten=True)

# Generate some sample data
data = tf.random.normal((2, 8, 8, 3))

# Apply selected pooling
output = select_pool(data)
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

# BigBird_attention

The `BigBird_attention` class implements BigBird, a sparse attention mechanism, which reduces the quadratic dependency of attention computation to linear. This implementation is based on the paper "Big Bird: Transformers for Longer Sequences" (https://arxiv.org/abs/2007.14062).

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`key_dim`** (int): Size of each attention head for query and key.
- **`input_size`** (int, optional): Size of the input.
- **`num_rand_blocks`** (int): Number of random blocks. Default is `3`.
- **`from_block_size`** (int): Block size of the query. Default is `64`.
- **`to_block_size`** (int): Block size of the key. Default is `64`.
- **`max_rand_mask_length`** (int): Maximum length for the random mask. Default is `MAX_SEQ_LEN`.
- **`weight_initializer`** (str): Initializer for the weights. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias. Default is `'zeros'`.
- **`use_bias`** (bool): If `True`, adds a bias term to the attention computation. Default is `True`.
- **`seed`** (int, optional): Seed for random number generation.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, query, value, key=None, attention_mask=None)`**: Applies BigBird sparse attention to the input `query`, `key`, and `value`.

  - **Parameters**:
    - **`query`**: Query tensor.
    - **`value`**: Value tensor.
    - **`key`** (optional): Key tensor. If not provided, `value` is used as the key.
    - **`attention_mask`** (optional): Mask tensor for attention computation.

  - **Returns**: Attention output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of BigBird_attention
bigbird_attn = nn.BigBird_attention(
    n_head=8,
    key_dim=64,
    input_size=128,
    num_rand_blocks=3,
    from_block_size=64,
    to_block_size=64,
    max_rand_mask_length=512,
    weight_initializer='Xavier',
    bias_initializer='zeros',
    use_bias=True,
    dtype='float32'
)

# Generate some sample data
query = tf.random.normal((2, 128, 128))
value = tf.random.normal((2, 128, 128))

# Apply BigBird attention
output = bigbird_attn(query, value)
```

# BigBird_masks

The `BigBird_masks` class creates attention masks for the BigBird attention mechanism, which are used to efficiently handle long sequences by reducing the complexity of the attention computation.

**Initialization Parameters**

- **`block_size`** (int): Size of the blocks used in the BigBird attention mechanism.

**Methods**

- **`__call__(self, data, mask)`**: Generates the attention masks required for BigBird attention.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`mask`**: Mask tensor indicating which elements should be attended to.

  - **Returns**: A list of masks `[band_mask, encoder_from_mask, encoder_to_mask, blocked_encoder_mask]` used in the BigBird attention mechanism.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the BigBird_masks class
bigbird_masks = nn.BigBird_masks(block_size=64)

# Generate some sample data and mask
data = tf.random.normal((2, 128, 128))
mask = tf.cast(tf.random.uniform((2, 128), maxval=2, dtype=tf.int32), tf.float32)

# Generate the BigBird attention masks
masks = bigbird_masks(data, mask)
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

# global_max_pool

The `global_max_pool` classes implement global max pooling operations for 1D, 2D, and 3D data. These layers reduce the input data by taking the maximum value over specified dimensions, helping to reduce the data size while preserving the most important features.

**Initialization Parameters**

- **`keepdims`** (bool): If `True`, retains reduced dimensions with length 1. Default is `False`.

**Methods**

- **`__call__(self, data)`**: Applies global max pooling to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor of shape `(batch_size, seq_length, feature_dim)` for 1D, `(batch_size, height, width, feature_dim)` for 2D, and `(batch_size, depth, height, width, feature_dim)` for 3D.
  - **Returns**: Tensor after applying global max pooling.

**Classes**

- **`global_max_pool1d`**: Applies global max pooling over 1D data.
- **`global_max_pool2d`**: Applies global max pooling over 2D data.
- **`global_max_pool3d`**: Applies global max pooling over 3D data.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# 1D Global Max Pooling
pool1d = nn.global_max_pool1d(keepdims=True)
data_1d = tf.random.normal((2, 10, 5))
output_1d = pool1d(data_1d)

# 2D Global Max Pooling
pool2d = nn.global_max_pool2d(keepdims=True)
data_2d = tf.random.normal((2, 8, 8, 5))
output_2d = pool2d(data_2d)

# 3D Global Max Pooling
pool3d = nn.global_max_pool3d(keepdims=True)
data_3d = tf.random.normal((2, 4, 4, 4, 5))
output_3d = pool3d(data_3d)
```

# grouped_query_attention

The `grouped_query_attention` class implements the grouped-query attention mechanism introduced by Ainslie et al. (2023). This mechanism improves the efficiency and scalability of attention layers in neural networks by grouping query, key, and value projections.

**Initialization Parameters**

- **`head_dim`** (int): Size of each attention head.
- **`num_query_heads`** (int): Number of query attention heads.
- **`num_key_value_heads`** (int): Number of key and value attention heads. Must be a divisor of `num_query_heads`.
- **`dropout_rate`** (float): Dropout probability. Default is `0.0`.
- **`use_bias`** (bool): If `True`, includes bias in dense layers. Default is `True`.
- **`weight_initializer`** (str): Initializer for dense layer kernels. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for dense layer biases. Default is `'zeros'`.
- **`query_shape`** (tuple, optional): Shape of the query tensor.
- **`value_shape`** (tuple, optional): Shape of the value tensor.
- **`key_shape`** (tuple, optional): Shape of the key tensor. If not provided, `value_shape` is used.

**Methods**

- **`__call__(self, query, value, key=None, query_mask=None, value_mask=None, key_mask=None, attention_mask=None, return_attention_scores=False, training=None, use_causal_mask=False)`**: Applies grouped-query attention to the input `query`, `key`, and `value`.

  - **Parameters**:
    - **`query`**: Query tensor of shape `(batch_size, target_seq_len, feature_dim)`.
    - **`value`**: Value tensor of shape `(batch_size, source_seq_len, feature_dim)`.
    - **`key`**: Key tensor of shape `(batch_size, source_seq_len, feature_dim)`. Defaults to `value`.
    - **`query_mask`** (optional): Mask for query tensor.
    - **`value_mask`** (optional): Mask for value tensor.
    - **`key_mask`** (optional): Mask for key tensor.
    - **`attention_mask`** (optional): Boolean mask for attention.
    - **`return_attention_scores`** (bool): If `True`, returns attention scores along with the output. Default is `False`.
    - **`training`** (bool): If `True`, applies dropout. Default is `None`.
    - **`use_causal_mask`** (bool): If `True`, applies a causal mask for decoder transformers. Default is `False`.

  - **Returns**: The attention output tensor and optionally the attention scores.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the grouped-query attention layer
gqa = nn.grouped_query_attention(
    head_dim=64,
    num_query_heads=8,
    num_key_value_heads=4,
    dropout_rate=0.1,
    use_bias=True
)

# Generate some sample data
query = tf.random.normal((2, 10, 128))
value = tf.random.normal((2, 20, 128))
key = tf.random.normal((2, 20, 128))

# Apply grouped-query attention
output = gqa(query, value, key)
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

# Linformer_self_attention

The `Linformer_self_attention` class implements the Linformer self-attention mechanism, which reduces the computational complexity of the traditional self-attention by projecting the sequence length dimension.

**Initialization Parameters**

- **`dim`** (int): Dimension of the input feature.
- **`seq_len`** (int): Sequence length of the input.
- **`k`** (int, optional): Reduced dimension for keys and values. Default is `256`.
- **`heads`** (int, optional): Number of attention heads. Default is `8`.
- **`dim_head`** (int, optional): Dimension of each attention head. If `None`, it is set to `dim // heads`.
- **`one_kv_head`** (bool, optional): If `True`, uses a single head for keys and values. Default is `False`.
- **`share_kv`** (bool, optional): If `True`, shares the same projection for keys and values. Default is `False`.
- **`dropout`** (float, optional): Dropout rate for attention weights. Default is `0.0`.
- **`dtype`** (str, optional): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, x, context=None, train_flag=True)`**: Applies the Linformer self-attention mechanism to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.
    - **`context`** (optional): Context tensor for cross-attention. If `None`, performs self-attention.
    - **`train_flag`** (bool, optional): Specifies whether the layer is in training mode.

  - **Returns**: Output tensor after applying Linformer self-attention.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the Linformer self-attention layer
linformer_sa = nn.Linformer_self_attention(dim=512, seq_len=128)

# Generate some sample data
data = tf.random.normal((2, 128, 512))

# Apply Linformer self-attention
output = linformer_sa(data)
```

# LoRALinear

The `LoRALinear` class implements the LoRA (Low-Rank Adaptation) for linear layers, which adapts pre-trained models by adding low-rank matrices to the weights.

**Initialization Parameters**

- **`input_dims`** (int): Dimension of the input feature.
- **`output_dims`** (int): Dimension of the output feature.
- **`lora_rank`** (int, optional): Rank of the low-rank matrices. Default is `8`.
- **`bias`** (bool, optional): If `True`, adds a bias term. Default is `False`.
- **`scale`** (float, optional): Scale factor for the low-rank updates. Default is `20.0`.

**Static Methods**

- **`from_linear(linear, rank=8)`**: Creates a `LoRALinear` instance from an existing linear layer.

  - **Parameters**:
    - **`linear`**: Existing linear layer.
    - **`rank`** (int, optional): Rank of the low-rank matrices. Default is `8`.

  - **Returns**: `LoRALinear` instance.

**Methods**

- **`to_linear(self)`**: Converts the `LoRALinear` layer back to a regular linear layer.

  - **Returns**: Regular linear layer with low-rank updates applied.

- **`__call__(self, data)`**: Applies the `LoRALinear` transformation to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: Output tensor after applying the `LoRALinear` transformation.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the LoRALinear layer
lora_linear = nn.LoRALinear(input_dims=128, output_dims=64)

# Generate some sample data
data = tf.random.normal((2, 128))

# Apply LoRALinear transformation
output = lora_linear(data)
```

# masked_lm

The `masked_lm` class implements a masked language model head, typically used in BERT-like models for predicting masked tokens.

**Initialization Parameters**

- **`vocab_size`** (int): Size of the vocabulary.
- **`hidden_size`** (int): Dimension of the hidden layer.
- **`input_size`** (int, optional): Dimension of the input feature.
- **`activation`** (str, optional): Activation function for the dense layer.
- **`initializer`** (str, optional): Initializer for the dense layer weights. Default is `'Xavier'`.
- **`output`** (str, optional): Output type, either `'logits'` or `'predictions'`. Default is `'logits'`.
- **`dtype`** (str, optional): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, sequence_data, embedding_table, masked_positions)`**: Applies the masked language model head to the input sequence data.

  - **Parameters**:
    - **`sequence_data`**: Input sequence tensor.
    - **`embedding_table`**: Embedding table tensor.
    - **`masked_positions`**: Positions of the masked tokens in the sequence.

  - **Returns**: Logits or predictions for the masked tokens.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the masked language model layer
mlm = nn.masked_lm(vocab_size=30522, hidden_size=768)

# Generate some sample data
sequence_data = tf.random.normal((2, 128, 768))
embedding_table = tf.random.normal((30522, 768))
masked_positions = tf.constant([[5, 15], [8, 23]])

# Apply masked language model
output = mlm(sequence_data, embedding_table, masked_positions)
```

# masked_softmax

The `masked_softmax` class performs a softmax operation with optional masking on a tensor, commonly used in attention mechanisms.

**Initialization Parameters**

- **`mask_expansion_axes`** (int, optional): Axes to expand the mask tensor for broadcasting.
- **`normalization_axes`** (tuple of int, optional): Axes on which to perform the softmax. Default is `(-1,)`.

**Methods**

- **`__call__(self, scores, mask=None)`**: Applies masked softmax to the input scores.

  - **Parameters**:
    - **`scores`**: Input score tensor.
    - **`mask`** (optional): Mask tensor.

  - **Returns**: Softmax output with masking applied.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the masked softmax layer
masked_sf = nn.masked_softmax()

# Generate some sample data
scores = tf.random.normal((2, 10, 10))
mask = tf.constant([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

# Apply masked softmax
output = masked_sf(scores, mask)
```

# matmul_with_margin

The `matmul_with_margin` class computes a dot product matrix given two encoded inputs with an optional margin.

**Initialization Parameters**

- **`logit_scale`** (float): The scaling factor for dot products during training. Default is `1.0`.
- **`logit_margin`** (float): The margin value between positive and negative examples during training. Default is `0.0`.

**Methods**

- **`__call__(self, left_encoded, right_encoded)`**: Computes the dot product matrix for the given encoded inputs.

  - **Parameters**:
    - **`left_encoded`** (tf.Tensor): The left encoded input tensor.
    - **`right_encoded`** (tf.Tensor): The right encoded input tensor.

  - **Returns**: A tuple containing the left and right logits.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the matmul_with_margin layer
mmwm = nn.matmul_with_margin(logit_scale=1.0, logit_margin=0.1)

# Generate some sample data
left_encoded = tf.random.normal((2, 5, 10))
right_encoded = tf.random.normal((2, 5, 10))

# Compute the dot product with margin
left_logits, right_logits = mmwm(left_encoded, right_encoded)
```

# max_pool1d

The `max_pool1d` class implements 1D max pooling.

**Initialization Parameters**

- **`ksize`** (int): Size of the max pooling window.
- **`strides`** (int): Stride of the max pooling window.
- **`padding`** (str): Padding algorithm to use, either `'VALID'` or `'SAME'`.

**Methods**

- **`__call__(self, data)`**: Applies 1D max pooling to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: The result of max pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the max_pool1d layer
mp1d = nn.max_pool1d(ksize=2, strides=2, padding='VALID')

# Generate some sample data
data = tf.random.normal((2, 10, 1))

# Apply max pooling
output = mp1d(data)
```

# max_pool2d

The `max_pool2d` class implements 2D max pooling.

**Initialization Parameters**

- **`ksize`** (int): Size of the max pooling window.
- **`strides`** (int): Stride of the max pooling window.
- **`padding`** (str): Padding algorithm to use, either `'VALID'` or `'SAME'`.

**Methods**

- **`__call__(self, data)`**: Applies 2D max pooling to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: The result of max pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the max_pool2d layer
mp2d = nn.max_pool2d(ksize=2, strides=2, padding='VALID')

# Generate some sample data
data = tf.random.normal((2, 10, 10, 3))

# Apply max pooling
output = mp2d(data)
```

# max_pool3d

The `max_pool3d` class implements 3D max pooling.

**Initialization Parameters**

- **`ksize`** (int): Size of the max pooling window.
- **`strides`** (int): Stride of the max pooling window.
- **`padding`** (str): Padding algorithm to use, either `'VALID'` or `'SAME'`.

**Methods**

- **`__call__(self, data)`**: Applies 3D max pooling to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: The result of max pooling.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the max_pool3d layer
mp3d = nn.max_pool3d(ksize=2, strides=2, padding='VALID')

# Generate some sample data
data = tf.random.normal((2, 10, 10, 10, 3))

# Apply max pooling
output = mp3d(data)
```

# maxout

The `maxout` class applies the Maxout operation to the input tensor.

**Initialization Parameters**

- **`num_units`** (int): Specifies how many features will remain after maxout in the specified `axis` dimension.
- **`axis`** (int): The dimension where max pooling will be performed. Default is `-1`.
- **`input_shape`** (tuple, optional): Shape of the input tensor.

**Methods**

- **`__call__(self, data)`**: Applies the Maxout operation to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: Tensor after applying the Maxout operation.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the maxout layer
maxout_layer = nn.maxout(num_units=4)

# Generate some sample data
data = tf.random.normal((2, 10, 8))

# Apply maxout
output = maxout_layer(data)
```

# Mlp

The `Mlp` class implements a Multi-Layer Perceptron (MLP) as used in Vision Transformer, MLP-Mixer, and related networks.

**Initialization Parameters**

- **`in_features`** (int): Size of each input sample.
- **`hidden_features`** (int, optional): Size of the hidden layer. Defaults to `in_features`.
- **`out_features`** (int, optional): Size of each output sample. Defaults to `in_features`.
- **`act_layer`** (function): Activation function to use. Default is `tf.nn.gelu`.
- **`norm_layer`** (function, optional): Normalization layer to use. Default is `None`.
- **`bias`** (bool): Whether to use bias in linear layers. Default is `True`.
- **`drop`** (float): Dropout probability. Default is `0.0`.
- **`use_conv`** (bool): Whether to use 1x1 convolutions instead of dense layers. Default is `False`.

**Methods**

- **`__call__(self, x)`**: Applies the MLP to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after applying the MLP.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the MLP layer
mlp = nn.Mlp(in_features=128)

# Generate some sample data
data = tf.random.normal((32, 128))

# Apply MLP
output = mlp(data)
```

# GluMlp

The `GluMlp` class implements a Gated Linear Unit (GLU) style MLP.

**Initialization Parameters**

- **`in_features`** (int): Size of each input sample.
- **`hidden_features`** (int, optional): Size of the hidden layer. Defaults to `in_features`.
- **`out_features`** (int, optional): Size of each output sample. Defaults to `in_features`.
- **`act_layer`** (function): Activation function to use. Default is `tf.nn.sigmoid`.
- **`norm_layer`** (function, optional): Normalization layer to use. Default is `None`.
- **`bias`** (bool): Whether to use bias in linear layers. Default is `True`.
- **`drop`** (float): Dropout probability. Default is `0.0`.
- **`use_conv`** (bool): Whether to use 1x1 convolutions instead of dense layers. Default is `False`.
- **`gate_last`** (bool): Whether to apply gating after the activation. Default is `True`.

**Methods**

- **`__call__(self, x)`**: Applies the GLU MLP to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after applying the GLU MLP.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the GLU MLP layer
glu_mlp = nn.GluMlp(in_features=128)

# Generate some sample data
data = tf.random.normal((32, 128))

# Apply GLU MLP
output = glu_mlp(data)
```

# SwiGLUPacked

The `SwiGLUPacked` class is a partial application of the `GluMlp` class with specific activation and gating.

**Initialization Parameters**

- **`in_features`** (int): Size of each input sample.
- **`hidden_features`** (int, optional): Size of the hidden layer. Defaults to `in_features`.
- **`out_features`** (int, optional): Size of each output sample. Defaults to `in_features`.
- **`act_layer`** (function): Activation function to use. Default is `tf.nn.silu`.
- **`norm_layer`** (function, optional): Normalization layer to use. Default is `None`.
- **`bias`** (bool): Whether to use bias in linear layers. Default is `True`.
- **`drop`** (float): Dropout probability. Default is `0.0`.
- **`use_conv`** (bool): Whether to use 1x1 convolutions instead of dense layers. Default is `False`.
- **`gate_last`** (bool): Whether to apply gating after the activation. Default is `False`.

**Methods**

- **`__call__(self, x)`**: Applies the SwiGLU Packed to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after applying the SwiGLU Packed.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the SwiGLUPacked layer
swiglu_packed = nn.SwiGLUPacked(in_features=128)

# Generate some sample data
data = tf.random.normal((32, 128))

# Apply SwiGLU Packed
output = swiglu_packed(data)
```

# SwiGLU

The `SwiGLU` class implements the SwiGLU operation.

**Initialization Parameters**

- **`in_features`** (int): Size of each input sample.
- **`hidden_features`** (int, optional): Size of the hidden layer. Defaults to `in_features`.
- **`out_features`** (int, optional): Size of each output sample. Defaults to `in_features`.
- **`act_layer`** (function): Activation function to use. Default is `tf.nn.silu`.
- **`norm_layer`** (function, optional): Normalization layer to use. Default is `None`.
- **`bias`** (bool): Whether to use bias in linear layers. Default is `True`.
- **`drop`** (float): Dropout probability. Default is `0.0`.

**Methods**

- **`__call__(self, x)`**: Applies the SwiGLU operation to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after applying the SwiGLU operation.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the SwiGLU layer
swiglu = nn.SwiGLU(in_features=128)

# Generate some sample data
data = tf.random.normal((32, 128))

# Apply SwiGLU
output = swiglu(data)
```

# GatedMlp

The `GatedMlp` class implements a Gated MLP as used in gMLP.

**Initialization Parameters**

- **`in_features`** (int): Size of each input sample.
- **`hidden_features`** (int, optional): Size of the hidden layer. Defaults to `in_features`.
- **`out_features`** (int, optional): Size of each output sample. Defaults to `in_features`.
- **`act_layer`** (function): Activation function to use. Default is `tf.nn.gelu`.
- **`norm_layer`** (function, optional): Normalization layer to use. Default is `None`.
- **`gate_layer`** (function, optional): Gate layer to use. Default is `None`.
- **`bias`** (bool): Whether to use bias in linear layers. Default is `True`.
- **`drop`** (float): Dropout probability. Default is `0.0`.

**Methods**

- **`__call__(self, x)`**: Applies the Gated MLP to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after applying the Gated MLP.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the Gated MLP layer
gated_mlp = nn.GatedMlp(in_features=128)

# Generate some sample data
data = tf.random.normal((32, 128))

# Apply Gated MLP
output = gated_mlp(data)
```

# ConvMlp

The `ConvMlp` class implements an MLP using 1x1 convolutions while preserving spatial dimensions.

**Initialization Parameters**

- **`in_features`** (int): Size of each input sample.
- **`hidden_features`** (int, optional): Size of the hidden layer. Defaults to `in_features`.
- **`out_features`** (int, optional): Size of each output sample. Defaults to `in_features`.
- **`act_layer`** (function): Activation function to use. Default is `tf.nn.relu`.
- **`norm_layer`** (function, optional): Normalization layer to use. Default is `None`.
- **`bias`** (bool): Whether to use bias in convolution layers. Default is `True`.
- **`drop`** (float): Dropout probability. Default is `0.0`.

**Methods**

- **`__call__(self, x)`**: Applies the ConvMlp to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after applying the ConvMlp.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the ConvMlp layer
conv_mlp = nn.ConvMlp(in_features=128)

# Generate some sample data
data = tf.random.normal

((32, 128, 32, 32))

# Apply ConvMlp
output = conv_mlp(data)
```

# GlobalResponseNormMlp

The `GlobalResponseNormMlp` class implements an MLP with Global Response Normalization.

**Initialization Parameters**

- **`in_features`** (int): Size of each input sample.
- **`hidden_features`** (int, optional): Size of the hidden layer. Defaults to `in_features`.
- **`out_features`** (int, optional): Size of each output sample. Defaults to `in_features`.
- **`act_layer`** (function): Activation function to use. Default is `tf.nn.gelu`.
- **`bias`** (bool): Whether to use bias in linear layers. Default is `True`.
- **`drop`** (float): Dropout probability. Default is `0.0`.
- **`use_conv`** (bool): Whether to use 1x1 convolutions instead of dense layers. Default is `False`.

**Methods**

- **`__call__(self, x)`**: Applies the Global Response Norm MLP to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after applying the Global Response Norm MLP.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the GlobalResponseNormMlp layer
grn_mlp = nn.GlobalResponseNormMlp(in_features=128)

# Generate some sample data
data = tf.random.normal((32, 128))

# Apply Global Response Norm MLP
output = grn_mlp(data)
```

# GlobalResponseNorm

The `GlobalResponseNorm` class implements a Global Response Normalization layer.

**Initialization Parameters**

- **`dim`** (int): Dimensionality of the input.
- **`eps`** (float): Small constant to avoid division by zero. Default is `1e-6`.
- **`channels_last`** (bool): If `True`, channels are the last dimension. Default is `True`.

**Methods**

- **`__call__(self, x)`**: Applies Global Response Normalization to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Normalized output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the Global Response Normalization layer
grn = nn.GlobalResponseNorm(dim=128)

# Generate some sample data
data = tf.random.normal((32, 128, 32, 32))

# Apply Global Response Normalization
output = grn(data)
```

# MoE_layer

The `MoE_layer` class implements a Sparse Mixture of Experts (MoE) layer with per-token routing.

**Initialization Parameters**

- **`experts`** (`feed_forward_experts`): Instance of `FeedForwardExperts`. Must have the same `num_experts` as the router.
- **`router`** (`MaskedRouter`): Instance of `MaskedRouter` to route the tokens to different experts.
- **`train_capacity_factor`** (float, optional): Scaling factor for expert token capacity during training. Default is `1.0`.
- **`eval_capacity_factor`** (float, optional): Scaling factor for expert token capacity during evaluation. Default is `1.0`.
- **`examples_per_group`** (float, optional): Number of examples to form a group for routing. Default is `1.0`.

**Methods**

- **`__call__(self, inputs, train_flag=True)`**: Applies the MoE layer to the input `inputs`.

  - **Parameters**:
    - **`inputs`** (`tf.Tensor`): Batch of input embeddings of shape `[batch_size, seq_length, hidden_dim]`.
    - **`train_flag`** (bool, optional): If `True`, applies dropout and jitter noise during training. Default is `True`.

  - **Returns**: Transformed inputs with the same shape as `inputs`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the MoE_layer
moe_layer = nn.MoE_layer(experts=my_experts, router=my_router)

# Generate some sample data
data = tf.random.normal((32, 128, 512))

# Apply MoE layer
output = moe_layer(data, train_flag=True)
```

# multi_cls_heads

The `multi_cls_heads` class implements multiple classification heads sharing the same pooling stem.

**Initialization Parameters**

- **`inner_dim`** (int): Dimensionality of the inner projection layer. If `0` or `None`, only the output projection layer is created.
- **`cls_list`** (list): List of numbers of classes for each classification head.
- **`input_size`** (int, optional): Size of the input.
- **`cls_token_idx`** (int, optional): Index inside the sequence to pool. Default is `0`.
- **`activation`** (str, optional): Activation function to use. Default is `"tanh"`.
- **`dropout_rate`** (float, optional): Dropout probability. Default is `0.0`.
- **`initializer`** (str, list, tuple): Initializer for dense layer kernels. Default is `"Xavier"`.
- **`dtype`** (str, optional): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, features, only_project=False)`**: Applies the multi-classification heads to the input `features`.

  - **Parameters**:
    - **`features`** (`tf.Tensor`): Rank-3 or rank-2 input tensor.
    - **`only_project`** (bool, optional): If `True`, returns the intermediate tensor before projecting to class logits. Default is `False`.

  - **Returns**: If `only_project` is `True`, returns a tensor with shape `[batch_size, hidden_size]`. Otherwise, returns a dictionary of tensors.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the multi_cls_heads
multi_heads = nn.multi_cls_heads(inner_dim=128, cls_list=[10, 20, 30])

# Generate some sample data
data = tf.random.normal((32, 128, 512))

# Apply multi-classification heads
output = multi_heads(data)
```

# multichannel_attention

The `multichannel_attention` class implements a Multi-channel Attention layer.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`key_dim`** (int): Dimensionality of the key.
- **`value_dim`** (int, optional): Dimensionality of the value. If `None`, defaults to `key_dim`.
- **`input_size`** (int, optional): Size of the input.
- **`dropout_rate`** (float, optional): Dropout probability. Default is `0.0`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias. Default is `'zeros'`.
- **`use_bias`** (bool, optional): If `True`, use bias in dense layers. Default is `True`.
- **`dtype`** (str, optional): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, query, value, key=None, context_attention_weights=None, attention_mask=None, train_flag=True)`**: Applies multi-channel attention to the input tensors.

  - **Parameters**:
    - **`query`** (`tf.Tensor`): Query tensor of shape `[B, T, dim]`.
    - **`value`** (`tf.Tensor`): Value tensor of shape `[B, A, S, dim]`.
    - **`key`** (`tf.Tensor`, optional): Key tensor of shape `[B, A, S, dim]`. Defaults to `value`.
    - **`context_attention_weights`** (`tf.Tensor`): Context weights of shape `[B, N, T, A]`.
    - **`attention_mask`** (`tf.Tensor`, optional): Boolean mask of shape `[B, T, S]` to prevent attention to certain positions.
    - **`train_flag`** (bool, optional): If `True`, apply dropout during training. Default is `True`.

  - **Returns**: Attention output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the multichannel_attention
multi_attention = nn.multichannel_attention(n_head=8, key_dim=64)

# Generate some sample data
query = tf.random.normal((32, 128, 512))
value = tf.random.normal((32, 4, 128, 512))

# Apply multi-channel attention
output = multi_attention(query, value)
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

# multiheadrelative_attention

The `multiheadrelative_attention` class implements a multi-head attention layer with relative attention and position encoding. This layer enhances the traditional multi-head attention mechanism by incorporating relative position information, which helps in better modeling of sequential data.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`key_dim`** (int): Dimensionality of the keys.
- **`input_size`** (int, optional): Size of the input. If not provided, it will be inferred from the input data.
- **`attention_axes`** (tuple of int, optional): Axes over which the attention is applied. Defaults to the last axis.
- **`dropout_rate`** (float): Dropout rate for the attention probabilities. Default is `0.0`.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight matrices. Default is `['VarianceScaling', 1.0, 'fan_in', 'truncated_normal']`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias parameters. Default is `'zeros'`.
- **`use_bias`** (bool): Whether to use bias parameters. Default is `True`.
- **`dtype`** (str): Data type of the parameters. Default is `'float32'`.

**Call Arguments**

- **`query`** (Tensor): Query tensor of shape `[B, T, dim]`.
- **`value`** (Tensor): Value tensor of shape `[B, S, dim]`.
- **`content_attention_bias`** (Tensor): Bias tensor for content-based attention of shape `[num_heads, dim]`.
- **`positional_attention_bias`** (Tensor): Bias tensor for position-based attention of shape `[num_heads, dim]`.
- **`key`** (Tensor, optional): Key tensor of shape `[B, S, dim]`. If not provided, `value` is used for both key and value.
- **`relative_position_encoding`** (Tensor): Relative positional encoding tensor of shape `[B, L, dim]`.
- **`segment_matrix`** (Tensor, optional): Segmentation IDs used in XLNet of shape `[B, S, S + M]`.
- **`segment_encoding`** (Tensor, optional): Segmentation encoding as used in XLNet of shape `[2, num_heads, dim]`.
- **`segment_attention_bias`** (Tensor, optional): Trainable bias parameter for segment-based attention of shape `[num_heads, dim]`.
- **`state`** (Tensor, optional): State tensor of shape `[B, M, E]` where M is the length of the state or memory.
- **`attention_mask`** (Tensor, optional): Boolean mask of shape `[B, T, S]` that prevents attention to certain positions.

**Methods**

**`__init__(self, n_head, key_dim, input_size=None, attention_axes=None, dropout_rate=0.0, weight_initializer=['VarianceScaling',1.0,'fan_in','truncated_normal'], bias_initializer='zeros', use_bias=True, dtype='float32')`**: Initializes the multi-head relative attention layer with the specified parameters.

**`build(self)`**: Builds the dense layers for query, key, value, output, and positional encodings if the input size is provided.

**`_masked_softmax(self, attention_scores, attention_mask=None)`**: Applies softmax to the attention scores, optionally using an attention mask to prevent attention to certain positions.

**`compute_attention(self, query, key, value, position, content_attention_bias, positional_attention_bias, segment_matrix=None, segment_encoding=None, segment_attention_bias=None, attention_mask=None)`**: Computes the multi-head relative attention over the inputs.

**`__call__(self, query, value, content_attention_bias, positional_attention_bias, key=None, relative_position_encoding=None, segment_matrix=None, segment_encoding=None, segment_attention_bias=None, state=None, attention_mask=None)`**: Applies the multi-head relative attention mechanism to the inputs.

**Usage Example**

```python
from Note import nn

attention_layer = nn.multiheadrelative_attention(
    n_head=8,
    key_dim=64,
    input_size=128,
    attention_axes=[1],
    dropout_rate=0.1
)

query = tf.random.normal([32, 10, 128])
value = tf.random.normal([32, 10, 128])
content_attention_bias = tf.random.normal([8, 64])
positional_attention_bias = tf.random.normal([8, 64])
relative_position_encoding = tf.random.normal([32, 20, 128])

output = attention_layer(
    query=query,
    value=value,
    content_attention_bias=content_attention_bias,
    positional_attention_bias=positional_attention_bias,
    relative_position_encoding=relative_position_encoding
)
```

# norm

The `norm` class is a preprocessing layer that normalizes continuous features by shifting and scaling inputs into a distribution centered around 0 with a standard deviation of 1.

**Initialization Parameters**

- **`input_shape`** (tuple, optional): Shape of the input data.
- **`axis`** (int, tuple of ints, None): The axis or axes along which to normalize. Defaults to `-1`.
- **`mean`** (float, optional): The mean value to use during normalization.
- **`variance`** (float, optional): The variance value to use during normalization.
- **`invert`** (bool): If `True`, applies the inverse transformation to the inputs. Default is `False`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`adapt(self, data)`**: Learns the mean and variance from the provided data and stores them as the layer's weights.
- **`__call__(self, data)`**: Normalizes the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor to be normalized.

  - **Returns**: Normalized output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the normalization layer
normalizer = nn.norm()

# Generate some sample data
data = tf.random.normal((2, 5, 10))

# Learn the normalization parameters
normalizer.adapt(data)

# Apply normalization
output = normalizer(data)
```

# perdimscale_attention

The `perdimscale_attention` class implements scaled dot-product attention with learned scales for individual dimensions, which can improve quality but might affect training stability.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`key_dim`** (int): Dimension of the key.
- **`value_dim`** (int, optional): Dimension of the value. Defaults to `key_dim`.
- **`input_size`** (int, optional): Size of the input.
- **`attention_axes`** (tuple of ints, optional): Axes over which the attention is applied.
- **`dropout_rate`** (float): Dropout rate. Default is `0.0`.
- **`weight_initializer`** (str): Initializer for weights. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for biases. Default is `'zeros'`.
- **`use_bias`** (bool): If `True`, adds bias to the dense layers. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, query, value, key=None, attention_mask=None, return_attention_scores=False, train_flag=True)`**: Applies attention mechanism to the inputs.

  - **Parameters**:
    - **`query`**: Query tensor.
    - **`value`**: Value tensor.
    - **`key`** (optional): Key tensor. If not provided, `value` is used as the key.
    - **`attention_mask`** (optional): Mask tensor for attention scores.
    - **`return_attention_scores`** (bool): If `True`, returns the attention scores. Default is `False`.
    - **`train_flag`** (bool): Specifies whether the layer is in training mode. Default is `True`.

  - **Returns**: Attention output tensor and optionally the attention scores.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the attention layer
attention = nn.perdimscale_attention(n_head=8, key_dim=64, input_size=128)

# Generate some sample data
query = tf.random.normal((2, 10, 128))
value = tf.random.normal((2, 10, 128))

# Apply attention
output = attention(query, value)
```

# permute

The `permute` class reorders the dimensions of the input according to a specified pattern.

**Initialization Parameters**

- **`dims`** (tuple of ints): Permutation pattern, indexing starts at 1.

**Methods**

- **`__call__(self, data)`**: Permutes the dimensions of the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor to be permuted.

  - **Returns**: Permuted output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the permute layer
permuter = nn.permute(dims=(2, 1))

# Generate some sample data
data = tf.random.normal((2, 5, 10))

# Apply permutation
output = permuter(data)
```

# position_embedding

The `position_embedding` class creates a positional embedding for the input sequence.

**Initialization Parameters**

- **`max_length`** (int): Maximum length of the sequence.
- **`input_size`** (int, optional): Size of the input.
- **`initializer`** (str, list, tuple): Initializer for the embedding weights. Default is `'Xavier'`.
- **`seq_axis`** (int): Axis of the input tensor where embeddings are added. Default is `1`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Adds positional embeddings to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor to which positional embeddings are added.

  - **Returns**: Tensor with added positional embeddings.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the positional embedding layer
pos_embedding = nn.position_embedding(max_length=50, input_size=128)

# Generate some sample data
data = tf.random.normal((2, 50, 128))

# Add positional embeddings
output = pos_embedding(data)
```

# PReLU

The `PReLU` class implements the Parametric Rectified Linear Unit activation function.

**Initialization Parameters**

- **`input_shape`** (tuple, optional): Shape of the input data.
- **`alpha_initializer`** (str, list, tuple): Initializer for the `alpha` weights. Default is `'zeros'`.
- **`shared_axes`** (list, tuple, optional): Axes along which to share learnable parameters.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the PReLU activation function to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor to which the PReLU activation function is applied.

  - **Returns**: Output tensor with PReLU activation applied.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the PReLU layer
prelu = nn.PReLU(input_shape=(None, 128))

# Generate some sample data
data = tf.random.normal((2, 128))

# Apply PReLU activation
output = prelu(data)
```

Here's the documentation for the classes `repeat_vector`, `reshape`, `reuse_multihead_attention`, and `RMSNorm` in the specified format for the README on GitHub. Only the imports from `Note` are included as requested.

```markdown
# Note Neural Network Layers

This repository contains custom neural network layers implemented for TensorFlow. Below is the documentation for each layer.

## repeat_vector

The `repeat_vector` class repeats the input tensor `n` times along a new axis.

**Initialization Parameters**

- **`n`** (int): The number of repetitions.

**Methods**

- **`__call__(self, data)`**: Repeats the input tensor.

  - **Parameters**:
    - **`data`**: Input 2D tensor of shape `(num_samples, features)`.
  - **Returns**: A 3D tensor of shape `(num_samples, n, features)`.

**Example Usage**

```python
from Note import nn

# Create an instance of the repeat vector layer
rv = nn.repeat_vector(3)

# Generate some sample data
data = tf.random.normal((2, 5))

# Apply repeat vector
output = rv(data)
```

# reshape

The `reshape` class reshapes the input tensor to the specified target shape.

**Initialization Parameters**

- **`target_shape`** (tuple of int): The desired output shape.

**Methods**

- **`__call__(self, data)`**: Reshapes the input tensor.

  - **Parameters**:
    - **`data`**: Input tensor.
  - **Returns**: A reshaped tensor.

**Example Usage**

```python
from Note import nn

# Create an instance of the reshape layer
reshape_layer = nn.reshape((10, 10))

# Generate some sample data
data = tf.random.normal((2, 100))

# Apply reshape
output = reshape_layer(data)
```

# reuse_multihead_attention

The `reuse_multihead_attention` class implements multi-head attention with optional reuse of attention heads.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`key_dim`** (int): Size of each attention head for query and key.
- **`value_dim`** (int, optional): Size of each attention head for value. Defaults to `key_dim`.
- **`input_size`** (int, optional): Size of the input.
- **`dropout`** (float): Dropout rate.
- **`reuse_attention`** (int): Number of heads to reuse. -1 for all heads.
- **`use_relative_pe`** (bool): Whether to use relative position bias.
- **`pe_max_seq_length`** (int): Maximum sequence length for relative position encodings.
- **`use_bias`** (bool): Whether to use bias in the dense layers.
- **`attention_axes`** (tuple of int, optional): Axes over which the attention is applied.
- **`weight_initializer`** (str, list, tuple): Initializer for the dense layer kernels.
- **`bias_initializer`** (str, list, tuple): Initializer for the dense layer biases.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, query, value, key=None, attention_mask=None, return_attention_scores=False, train_flag=True, reuse_attention_scores=None)`**: Applies multi-head attention.

  - **Parameters**:
    - **`query`**: Query tensor of shape `(B, T, dim)`.
    - **`value`**: Value tensor of shape `(B, S, dim)`.
    - **`key`** (optional): Key tensor of shape `(B, S, dim)`. Defaults to `value`.
    - **`attention_mask`** (optional): Boolean mask tensor.
    - **`return_attention_scores`** (bool, optional): Whether to return attention scores.
    - **`train_flag`** (bool, optional): Training mode flag.
    - **`reuse_attention_scores`** (optional): Precomputed attention scores for reuse.
  - **Returns**: Attention output tensor, and optionally attention scores.

**Example Usage**

```python
from Note import nn

# Create an instance of the multi-head attention layer
mha = nn.reuse_multihead_attention(n_head=8, key_dim=64, input_size=128)

# Generate some sample data
query = tf.random.normal((2, 10, 128))
value = tf.random.normal((2, 10, 128))

# Apply multi-head attention
output = mha(query, value)
```

# RMSNorm

The `RMSNorm` class implements Root Mean Square Layer Normalization.

**Initialization Parameters**

- **`dims`** (int): Dimensionality of the input.
- **`eps`** (float): Small constant to avoid division by zero. Default is `1e-6`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, x)`**: Applies RMS normalization.

  - **Parameters**:
    - **`x`**: Input tensor.
  - **Returns**: Normalized output tensor.

**Example Usage**

```python
from Note import nn

# Create an instance of the RMSNorm layer
rms_norm = nn.RMSNorm(dims=128)

# Generate some sample data
data = tf.random.normal((2, 10, 128))

# Apply RMS normalization
output = rms_norm(data)
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

# RoPE (Rotary Positional Encoding)

The `RoPE` class implements Rotary Positional Encoding, a technique used to encode positional information in transformer models.

**Initialization Parameters**

- **`dims`** (int): Dimensionality of the input vectors.
- **`traditional`** (bool, optional): If `True`, use traditional positional encoding. Default is `False`.
- **`base`** (float, optional): Base value used for frequency calculations. Default is `10000`.

**Methods**

- **`__call__(self, x, offset=0)`**: Applies Rotary Positional Encoding to the input `x`.

  - **Parameters**:
    - **`x`** (tf.Tensor): Input tensor to encode.
    - **`offset`** (int, optional): Positional offset for encoding. Default is `0`.
  - **Returns**: Tensor with positional encoding applied.

- **`create_cos_sin_theta(N, D, offset=0, base=10000, dtype=tf.float32)`**: Static method to create cosine and sine theta values for encoding.
  - **Parameters**:
    - **`N`** (int): Number of positions.
    - **`D`** (int): Dimensionality.
    - **`offset`** (int, optional): Offset for positional encoding. Default is `0`.
    - **`base`** (float, optional): Base value for frequency calculation. Default is `10000`.
    - **`dtype`** (tf.DType, optional): Data type for the encoding. Default is `tf.float32`.
  - **Returns**: Tuple of cosine and sine theta values.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of RoPE
rope = nn.RoPE(dims=64)

# Generate some sample data
data = tf.random.normal((2, 5, 64))

# Apply RoPE
output = rope(data)
```

# router

The `router` class provides an abstract base class for implementing token routing in Mixture of Experts (MoE) models.

**Initialization Parameters**

- **`num_experts`** (int): Number of experts.
- **`input_size`** (int, optional): Size of the input.
- **`jitter_noise`** (float, optional): Amplitude of jitter noise applied to router logits. Default is `0.0`.
- **`use_bias`** (bool, optional): Whether to use bias in the router weights. Default is `True`.
- **`kernel_initializer`** (str, list, tuple): Initializer for kernel weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for bias weights. Default is `'zeros'`.
- **`router_z_loss_weight`** (float, optional): Weight for router z-loss. Default is `0.0`.
- **`export_metrics`** (bool, optional): Whether to export metrics. Default is `True`.

**Methods**

- **`__call__(self, inputs, expert_capacity, train_flag=True)`**: Computes dispatch and combine arrays for routing to experts.

  - **Parameters**:
    - **`inputs`** (tf.Tensor): Input tensor.
    - **`expert_capacity`** (int): Capacity of each expert.
    - **`train_flag`** (bool, optional): Whether to apply jitter noise during routing. Default is `True`.
  - **Returns**: Routing instructions for the experts.

- **`_compute_router_probabilities(self, inputs, apply_jitter)`**: Computes router probabilities from input tokens.
  - **Parameters**:
    - **`inputs`** (tf.Tensor): Input tensor.
    - **`apply_jitter`** (bool): Whether to apply jitter noise.
  - **Returns**: Router probabilities and raw logits.

- **`_compute_routing_instructions(self, router_probs, expert_capacity)`**: Abstract method to compute routing instructions.
  - **Parameters**:
    - **`router_probs`** (tf.Tensor): Router probabilities.
    - **`expert_capacity`** (int): Capacity of each expert.
  - **Returns**: Routing instructions.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the router
r = nn.router(num_experts=4, input_size=128)

# Generate some sample data
data = tf.random.normal((2, 5, 128))

# Apply routing
output = r(data, expert_capacity=2)
```

# select_topk

The `select_topk` class selects top-k tokens according to importance, optionally with random selection.

**Initialization Parameters**

- **`top_k`** (int, optional): Number of top-k tokens to select.
- **`random_k`** (int, optional): Number of random tokens to select.

**Methods**

- **`__call__(self, data)`**: Selects top-k and/or random tokens from the input `data`.

  - **Parameters**:
    - **`data`** (tf.Tensor): Input tensor from which to select tokens.
  - **Returns**: Indices of selected and not-selected tokens.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of select_topk
selector = nn.select_topk(top_k=3, random_k=2)

# Generate some sample data
data = tf.random.normal((2, 5))

# Apply selection
selected, not_selected = selector(data)
```

# self_attention_mask

The `self_attention_mask` class creates a 3D attention mask from a 2D tensor mask.

**Methods**

- **`__call__(self, inputs, to_mask=None)`**: Creates the attention mask.

  - **Parameters**:
    - **`inputs`** (list or tf.Tensor): Input tensors. If `list`, the first element is the from_tensor and the second is the to_mask.
    - **`to_mask`** (tf.Tensor, optional): Mask tensor.
  - **Returns**: Attention mask tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of self_attention_mask
mask = nn.self_attention_mask()

# Generate some sample data
from_tensor = tf.random.normal((2, 5, 10))
to_mask = tf.ones((2, 5))

# Create the attention mask
attention_mask = mask([from_tensor, to_mask])
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

# separable_conv2d

The `separable_conv2d` class implements a separable convolutional layer, which performs depthwise convolution followed by pointwise convolution, reducing computational cost and model size.

**Initialization Parameters**

- **`filters`** (int): Number of output filters.
- **`kernel_size`** (tuple): Size of the convolutional kernel.
- **`depth_multiplier`** (int): Multiplier for the depthwise convolution output channels.
- **`input_size`** (int, optional): Number of input channels.
- **`strides`** (list): Stride size for the convolution. Default is `[1,1]`.
- **`padding`** (str): Padding method, either `'VALID'` or `'SAME'`. Default is `'VALID'`.
- **`data_format`** (str): Data format, either `'NHWC'` or `'NCHW'`. Default is `'NHWC'`.
- **`dilations`** (list, optional): Dilation rate for dilated convolution.
- **`weight_initializer`** (str, list, tuple): Initializer for the weight matrices. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the bias vector. Default is `'zeros'`.
- **`activation`** (str, optional): Activation function to apply.
- **`use_bias`** (bool): Whether to use a bias vector. Default is `True`.
- **`trainable`** (bool): Whether the layer is trainable. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the separable convolution to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: Output tensor after applying depthwise and pointwise convolution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the separable convolutional layer
conv = nn.separable_conv2d(filters=32, kernel_size=(3, 3), depth_multiplier=1, input_size=3)

# Generate some sample data
data = tf.random.normal((2, 64, 64, 3))

# Apply separable convolution
output = conv(data)
```

# spatial_dropout1d

The `spatial_dropout1d` class implements spatial dropout for 1D inputs, setting entire feature maps to zero.

**Initialization Parameters**

- **`rate`** (float): Fraction of the input units to drop. Between 0 and 1.
- **`seed`** (int): Random seed for reproducibility. Default is `7`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Applies spatial dropout to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`train_flag`** (bool): Whether to apply dropout. Default is `True`.

  - **Returns**: Output tensor after applying dropout.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the spatial dropout layer
dropout = nn.spatial_dropout1d(rate=0.5)

# Generate some sample data
data = tf.random.normal((2, 64, 3))

# Apply spatial dropout
output = dropout(data, train_flag=True)
```

# spatial_dropout2d

The `spatial_dropout2d` class implements spatial dropout for 2D inputs, setting entire feature maps to zero.

**Initialization Parameters**

- **`rate`** (float): Fraction of the input units to drop. Between 0 and 1.
- **`seed`** (int): Random seed for reproducibility. Default is `7`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Applies spatial dropout to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`train_flag`** (bool): Whether to apply dropout. Default is `True`.

  - **Returns**: Output tensor after applying dropout.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the spatial dropout layer
dropout = nn.spatial_dropout2d(rate=0.5)

# Generate some sample data
data = tf.random.normal((2, 64, 64, 3))

# Apply spatial dropout
output = dropout(data, train_flag=True)
```

# spatial_dropout3d

The `spatial_dropout3d` class implements spatial dropout for 3D inputs, setting entire feature maps to zero.

**Initialization Parameters**

- **`rate`** (float): Fraction of the input units to drop. Between 0 and 1.
- **`seed`** (int): Random seed for reproducibility. Default is `7`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Applies spatial dropout to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`train_flag`** (bool): Whether to apply dropout. Default is `True`.

  - **Returns**: Output tensor after applying dropout.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the spatial dropout layer
dropout = nn.spatial_dropout3d(rate=0.5)

# Generate some sample data
data = tf.random.normal((2, 10, 64, 64, 3))

# Apply spatial dropout
output = dropout(data, train_flag=True)
```

# spectral_norm

The `spectral_norm` class performs spectral normalization on the weights of a target layer, which helps to stabilize training by controlling the Lipschitz constant of the weights.

**Initialization Parameters**

- **`layer`** (Layer): Target layer to be normalized.
- **`power_iterations`** (int): Number of iterations during normalization. Default is `1`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Applies spectral normalization to the target layer's weights.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`train_flag`** (bool): Whether to apply spectral normalization. Default is `True`.

  - **Returns**: Output tensor from the target layer.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of a target layer (e.g., Conv2D)
conv_layer = nn.conv2d(filters=32, kernel_size=(3, 3), input_size=3)

# Wrap the target layer with spectral normalization
sn_layer = nn.spectral_norm(layer=conv_layer)

# Generate some sample data
data = tf.random.normal((2, 64, 64, 3))

# Apply spectral normalization and get the output
output = sn_layer(data, train_flag=True)
```

# SEModule

The `SEModule` class implements a Squeeze-and-Excitation module, which adaptively recalibrates channel-wise feature responses.

**Initialization Parameters**

- **`channels`** (int): Number of input channels.
- **`rd_ratio`** (float): Reduction ratio for the bottleneck. Default is `1/16`.
- **`rd_channels`** (int, optional): Number of reduction channels.
- **`rd_divisor`** (int): Divisor to ensure reduction channels are divisible. Default is `8`.
- **`add_maxpool`** (bool): Whether to add global max pooling. Default is `False`.
- **`bias`** (bool): Whether to use bias in convolutional layers. Default is `True`.
- **`act_layer`** (function): Activation function. Default is `tf.nn.relu`.
- **`norm_layer`** (Layer, optional): Normalization layer.
- **`gate_layer`** (function): Gate function for excitation. Default is `tf.nn.sigmoid`.

**Methods**

- **`__call__(self, x)`**: Applies the Squeeze-and-Excitation module to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Output tensor after Squeeze-and-Excitation.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the SE module
se_module = nn.SEModule(channels=64)

# Generate some sample data
data = tf.random.normal((2, 64, 64, 64))

# Apply the SE module
output = se_module(data)
```

# EffectiveSEModule

The `EffectiveSEModule` class implements an effective Squeeze-and-Excitation module as described in "CenterMask: Real-Time Anchor-Free Instance Segmentation".

**Initialization Parameters**

- **`channels`** (int): Number of input channels.
- **`add_maxpool`** (bool): Whether to add global max pooling. Default is `False`.
- **`gate_layer`** (function): Gate function for excitation. Default is `tf.nn.hard_sigmoid`.

**Methods**

- **`__call__(self, x)`**: Applies the effective Squeeze-and-Excitation module to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns

**: Output tensor after effective Squeeze-and-Excitation.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the effective SE module
effective_se_module = nn.EffectiveSEModule(channels=64)

# Generate some sample data
data = tf.random.normal((2, 64, 64, 64))

# Apply the effective SE module
output = effective_se_module(data)
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

# SwitchLinear

The `SwitchLinear` class implements a linear layer with multiple experts, where each expert has its own set of weights and biases. This is useful in mixture-of-experts models.

**Initialization Parameters**

- **`input_dims`** (int): Number of input dimensions.
- **`output_dims`** (int): Number of output dimensions.
- **`num_experts`** (int): Number of experts.
- **`bias`** (bool): Whether to include a bias term. Default is `True`.

**Methods**

- **`__call__(self, x, indices)`**: Applies the layer to the input `x` using the experts specified by `indices`.

  - **Parameters**:
    - **`x`**: Input tensor.
    - **`indices`**: Indices of the experts to use for each sample in the batch.

  - **Returns**: Output tensor after applying the selected experts.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the SwitchLinear layer
switch_linear = nn.SwitchLinear(input_dims=10, output_dims=20, num_experts=4)

# Generate some sample data
data = tf.random.normal((2, 10))
indices = tf.constant([0, 1])

# Apply SwitchLinear
output = switch_linear(data, indices)
```

# SwitchGLU

The `SwitchGLU` class implements a Gated Linear Unit (GLU) with multiple experts, where each expert has its own set of weights. This is useful in mixture-of-experts models.

**Initialization Parameters**

- **`input_dims`** (int): Number of input dimensions.
- **`hidden_dims`** (int): Number of hidden dimensions.
- **`num_experts`** (int): Number of experts.
- **`activation`** (function): Activation function to apply. Default is `tf.nn.silu`.
- **`bias`** (bool): Whether to include a bias term. Default is `False`.

**Methods**

- **`__call__(self, x, indices)`**: Applies the layer to the input `x` using the experts specified by `indices`.

  - **Parameters**:
    - **`x`**: Input tensor.
    - **`indices`**: Indices of the experts to use for each sample in the batch.

  - **Returns**: Output tensor after applying the selected experts.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the SwitchGLU layer
switch_glu = nn.SwitchGLU(input_dims=10, hidden_dims=20, num_experts=4)

# Generate some sample data
data = tf.random.normal((2, 10))
indices = tf.constant([0, 1])

# Apply SwitchGLU
output = switch_glu(data, indices)
```

# talking_heads_attention

The `talking_heads_attention` class implements a form of multi-head attention with linear projections applied to the attention scores before and after the softmax operation.

**Initialization Parameters**

- **`attention_axes`** (tuple): Axes over which to apply attention.
- **`dropout_rate`** (float): Dropout rate to apply to the attention scores. Default is `0.0`.
- **`initializer`** (str, list, tuple): Initializer for the attention weights. Default is `'Xavier'`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, query_tensor, key_tensor, value_tensor, attention_mask=None, train_flag=True)`**: Applies the attention mechanism to the input tensors.

  - **Parameters**:
    - **`query_tensor`**: Query tensor.
    - **`key_tensor`**: Key tensor.
    - **`value_tensor`**: Value tensor.
    - **`attention_mask`** (tensor, optional): Mask to apply to the attention scores.
    - **`train_flag`** (bool): Whether the layer is in training mode. Default is `True`.

  - **Returns**: Tuple of the attention output and the attention scores.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the talking_heads_attention layer
attention_layer = nn.talking_heads_attention()

# Generate some sample data
query = tf.random.normal((2, 5, 10))
key = tf.random.normal((2, 5, 10))
value = tf.random.normal((2, 5, 10))

# Apply attention
output, scores = attention_layer(query, key, value)
```

# thresholded_relu

The `thresholded_relu` class is a deprecated activation function that applies a thresholded ReLU activation.

**Initialization Parameters**

- **`theta`** (float): Threshold value. Default is `1.0`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the thresholded ReLU activation to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: Output tensor after applying the activation.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the thresholded ReLU layer
threshold_relu = nn.thresholded_relu(theta=1.0)

# Generate some sample data
data = tf.random.normal((2, 10))

# Apply thresholded ReLU
output = threshold_relu(data)
```

# TLU

The `TLU` (Thresholded Linear Unit) class implements an activation function similar to ReLU but with a learned threshold, beneficial for models using Filter Response Normalization (FRN).

**Initialization Parameters**

- **`input_shape`** (tuple, optional): Shape of the input.
- **`affine`** (bool): Whether to make it TLU-Affine, which has the form `max(x, α*x + τ)`. Default is `False`.
- **`tau_initializer`** (str, list, tuple): Initializer for the tau parameter. Default is `'zeros'`.
- **`alpha_initializer`** (str, list, tuple): Initializer for the alpha parameter. Default is `'zeros'`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Applies the TLU activation to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: Output tensor after applying the activation.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the TLU layer
tlu = nn.TLU(input_shape=(None, 10))

# Generate some sample data
data = tf.random.normal((2, 10))

# Apply TLU
output = tlu(data)
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

# two_stream_relative_attention

The `two_stream_relative_attention` class implements the two-stream relative self-attention mechanism used in XLNet. This attention mechanism involves two streams of attention: the content stream and the query stream. The content stream captures both the context and content, while the query stream focuses on contextual information and position, excluding the content.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`key_dim`** (int): Size of each attention head for query, key, and value.
- **`input_size`** (int, optional): Size of the input dimension.
- **`attention_axes`** (tuple, optional): Axes over which to apply attention.
- **`dropout_rate`** (float): Dropout rate to apply after attention. Default is `0.0`.
- **`weight_initializer`** (str, list, tuple): Initializer for the attention weights. Default is `'Xavier'`.
- **`bias_initializer`** (str, list, tuple): Initializer for the attention bias. Default is `'zeros'`.
- **`use_bias`** (bool): Whether to use a bias in the attention calculation. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, content_stream, content_attention_bias, positional_attention_bias, query_stream, relative_position_encoding, target_mapping=None, segment_matrix=None, segment_encoding=None, segment_attention_bias=None, state=None, content_attention_mask=None, query_attention_mask=None)`**: Computes the two-stream relative attention.

  - **Parameters**:
    - **`content_stream`**: Tensor of shape `[B, T, dim]` representing the content stream.
    - **`content_attention_bias`**: Bias tensor for content-based attention.
    - **`positional_attention_bias`**: Bias tensor for position-based attention.
    - **`query_stream`**: Tensor of shape `[B, P, dim]` representing the query stream.
    - **`relative_position_encoding`**: Tensor of relative positional encodings.
    - **`target_mapping`** (optional): Tensor for target mapping in partial prediction.
    - **`segment_matrix`** (optional): Tensor representing segmentation IDs.
    - **`segment_encoding`** (optional): Tensor representing the segmentation encoding.
    - **`segment_attention_bias`** (optional): Bias tensor for segment-based attention.
    - **`state`** (optional): Tensor representing the state or memory.
    - **`content_attention_mask`** (optional): Boolean mask for content attention.
    - **`query_attention_mask`** (optional): Boolean mask for query attention.

  - **Returns**: Tuple containing the content attention output and the query attention output, both of shape `[B, T, E]`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the two-stream relative attention layer
attention_layer = nn.two_stream_relative_attention(n_head=8, key_dim=64, input_size=128)

# Generate some sample data
content_stream = tf.random.normal((2, 10, 128))
query_stream = tf.random.normal((2, 10, 128))
relative_position_encoding = tf.random.normal((2, 10, 64))
content_attention_bias = tf.random.normal((8, 64))
positional_attention_bias = tf.random.normal((8, 64))

# Apply two-stream relative attention
content_attention_output, query_attention_output = attention_layer(
    content_stream=content_stream,
    content_attention_bias=content_attention_bias,
    positional_attention_bias=positional_attention_bias,
    query_stream=query_stream,
    relative_position_encoding=relative_position_encoding
)
```

# unfold

The `unfold` class extracts patches from the input tensor and flattens them, useful for operations like convolutional layers in neural networks.

**Initialization Parameters**

- **`kernel`** (int): Size of the extraction kernel.
- **`stride`** (int): Stride for the sliding window. Default is `1`.
- **`padding`** (int): Amount of padding to add to the input. Default is `0`.
- **`dilation`** (int): Dilation rate for the sliding window. Default is `1`.

**Methods**

- **`__call__(self, x)`**: Applies the unfolding operation to the input `x`.

  - **Parameters**:
    - **`x`**: Input tensor.

  - **Returns**: Tensor with extracted patches.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the unfold layer
unfold = nn.unfold(kernel=3, stride=1, padding=1)

# Generate some sample data
data = tf.random.normal((2, 5, 5, 3))

# Apply unfolding
output = unfold(data)
```

# unit_norm

The `unit_norm` class normalizes the input tensor along specified axes so that each input in the batch has a unit L2 norm.

**Initialization Parameters**

- **`axis`** (int or list/tuple of ints): The axis or axes to normalize across. Default is `-1`.

**Methods**

- **`__call__(self, inputs)`**: Normalizes the input tensor.

  - **Parameters**:
    - **`inputs`**: Input tensor.

  - **Returns**: Normalized tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the unit normalization layer
unit_norm = nn.unit_norm(axis=-1)

# Generate some sample data
data = tf.random.normal((2, 3))

# Apply unit normalization
output = unit_norm(data)
```

# up_sampling1d

The `up_sampling1d` class repeats each time step of the input tensor along the temporal axis.

**Initialization Parameters**

- **`size`** (int): The number of repetitions for each time step.

**Methods**

- **`__call__(self, inputs)`**: Applies 1D up-sampling to the input `inputs`.

  - **Parameters**:
    - **`inputs`**: Input tensor.

  - **Returns**: Up-sampled tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the up-sampling 1D layer
up_sampling1d = nn.up_sampling1d(size=2)

# Generate some sample data
data = tf.random.normal((2, 5, 3))

# Apply 1D up-sampling
output = up_sampling1d(data)
```

# up_sampling2d

The `up_sampling2d` class repeats each spatial dimension of the input tensor along the height and width axes.

**Initialization Parameters**

- **`size`** (int or tuple of ints): The number of repetitions for each spatial dimension. If an integer, the same value is used for both dimensions.

**Methods**

- **`__call__(self, inputs)`**: Applies 2D up-sampling to the input `inputs`.

  - **Parameters**:
    - **`inputs`**: Input tensor.

  - **Returns**: Up-sampled tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the up-sampling 2D layer
up_sampling2d = nn.up_sampling2d(size=(2, 2))

# Generate some sample data
data = tf.random.normal((2, 5, 5, 3))

# Apply 2D up-sampling
output = up_sampling2d(data)
```

# up_sampling3d

The `up_sampling3d` class repeats each spatial dimension of the input tensor along the depth, height, and width axes.

**Initialization Parameters**

- **`size`** (int or tuple of ints): The number of repetitions for each spatial dimension. If an integer, the same value is used for all dimensions.

**Methods**

- **`__call__(self, inputs)`**: Applies 3D up-sampling to the input `inputs`.

  - **Parameters**:
    - **`inputs`**: Input tensor.

  - **Returns**: Up-sampled tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the up-sampling 3D layer
up_sampling3d = nn.up_sampling3d(size=2)

# Generate some sample data
data = tf.random.normal((2, 5, 5, 5, 3))

# Apply 3D up-sampling
output = up_sampling3d(data)
```

# vector_quantizer

The `vector_quantizer` class implements vector quantization, a technique often used in neural network-based image and signal processing.

**Initialization Parameters**

- **`embedding_dim`** (int): Dimension of the embedding vectors.
- **`num_embeddings`** (int): Number of embedding vectors.
- **`commitment_cost`** (float): Commitment cost used in the loss term.
- **`dtype`** (str): Data type for the embeddings. Default is `'float32'`.

**Methods**

- **`__call__(self, data, is_training)`**: Applies vector quantization to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`is_training`** (bool): Indicates whether the layer is in training mode.

  - **Returns**: Dictionary containing quantized tensor, loss, perplexity, encodings, encoding indices, and distances.

- **`quantize(self, encoding_indices)`**: Converts encoding indices to quantized embeddings.

  - **Parameters**:
    - **`encoding_indices`**: Indices of the embeddings.

  - **Returns**: Quantized embeddings tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the vector quantizer layer
vector_quantizer = nn.vector_quantizer(embedding_dim=64, num_embeddings=512, commitment_cost=0.25)

# Generate some sample data
data = tf.random.normal((2, 10, 64))

# Apply vector quantization
output = vector_quantizer(data, is_training=True)
```

# PatchEmbed

The `PatchEmbed` class converts 2D images to patch embeddings, typically used in vision transformers.

**Initialization Parameters**

- **`img_size`** (int, optional): Size of the input image. Default is `224`.
- **`patch_size`** (int): Size of the patch. Default is `16`.
- **`in_chans`** (int): Number of input channels. Default is `3`.
- **`embed_dim`** (int): Dimension of the embedding. Default is `768`.
- **`flatten`** (bool): If `True`, flatten the output. Default is `True`.
- **`bias`** (bool): If `True`, add bias in the convolution layer. Default is `True`.

**Methods**

- **`__call__(self, x)`**: Converts the input image to patch embeddings.

  - **Parameters**:
    - **`x`**: Input tensor of shape `(batch_size, height, width, channels)`.

  - **Returns**: Patch embeddings tensor.

**Example Usage**

```python
from Note import nn

# Create an instance of the PatchEmbed layer
patch_embed = nn.PatchEmbed(img_size=224, patch_size=16)

# Generate some sample data
data = tf.random.normal((2, 224, 224, 3))

# Apply patch embedding
output = patch_embed(data)
```

# Attention

The `Attention` class implements multi-head self-attention, used in transformer models.

**Initialization Parameters**

- **`dim`** (int): Input dimension.
- **`num_heads`** (int): Number of attention heads. Default is `8`.
- **`qkv_bias`** (bool): If `True`, add bias in the query, key, value projections. Default is `False`.
- **`qk_norm`** (bool): If `True`, apply normalization to the query and key. Default is `False`.
- **`attn_drop`** (float): Dropout rate for attention weights. Default is `0.0`.
- **`proj_drop`** (float): Dropout rate for the output projection. Default is `0.0`.
- **`norm_layer`** (class): Normalization layer to use. Default is `nn.layer_norm`.
- **`use_fused_attn`** (bool): If `True`, use fused attention for efficiency. Default is `True`.

**Methods**

- **`__call__(self, x)`**: Applies multi-head self-attention to the input.

  - **Parameters**:
    - **`x`**: Input tensor of shape `(batch_size, seq_length, dim)`.

  - **Returns**: Output tensor after applying attention.

**Example Usage**

```python
from Note import nn

# Create an instance of the Attention layer
attention = nn.Attention(dim=768, num_heads=8)

# Generate some sample data
data = tf.random.normal((2, 10, 768))

# Apply attention
output = attention(data)
```

# Block

The `Block` class implements a transformer block with multi-head attention and MLP.

**Initialization Parameters**

- **`dim`** (int): Input dimension.
- **`num_heads`** (int): Number of attention heads.
- **`mlp_ratio`** (float): Ratio of hidden dimension in the MLP. Default is `4.0`.
- **`qkv_bias`** (bool): If `True`, add bias in the query, key, value projections. Default is `False`.
- **`qk_norm`** (bool): If `True`, apply normalization to the query and key. Default is `False`.
- **`proj_drop`** (float): Dropout rate for the output projection. Default is `0.0`.
- **`attn_drop`** (float): Dropout rate for attention weights. Default is `0.0`.
- **`init_values`** (float, optional): Initialization value for LayerScale. Default is `None`.
- **`drop_path`** (float): Dropout rate for stochastic depth. Default is `0.0`.
- **`act_layer`** (class): Activation function to use. Default is `tf.nn.gelu`.
- **`norm_layer`** (class): Normalization layer to use. Default is `nn.layer_norm`.
- **`mlp_layer`** (class): MLP layer to use. Default is `nn.Mlp`.

**Methods**

- **`__call__(self, x)`**: Applies the transformer block to the input.

  - **Parameters**:
    - **`x`**: Input tensor of shape `(batch_size, seq_length, dim)`.

  - **Returns**: Output tensor after applying the transformer block.

**Example Usage**

```python
from Note import nn

# Create an instance of the Block layer
block = nn.Block(dim=768, num_heads=8)

# Generate some sample data
data = tf.random.normal((2, 10, 768))

# Apply the transformer block
output = block(data)
```

# voting_attention

The `voting_attention` class implements a voting attention mechanism.

**Initialization Parameters**

- **`n_head`** (int): Number of attention heads.
- **`head_size`** (int): Size of each attention head.
- **`input_size`** (int, optional): Size of the input. Default is `None`.
- **`weight_initializer`** (str, list, tuple): Initializer for the dense layer weights. Default is `"Xavier"`.
- **`bias_initializer`** (str, list, tuple): Initializer for the dense layer biases. Default is `"zeros"`.
- **`use_bias`** (bool): If `True`, add bias in the dense layers. Default is `True`.
- **`dtype`** (str): Data type for the layer. Default is `'float32'`.

**Methods**

- **`__call__(self, encoder_outputs, doc_attention_mask)`**: Applies voting attention to the encoder outputs.

  - **Parameters**:
    - **`encoder_outputs`**: Encoder output tensor of shape `(batch_size, num_docs, seq_length, dim)`.
    - **`doc_attention_mask`**: Mask tensor for the attention mechanism.

  - **Returns**: Attention probabilities tensor.

**Example Usage**

```python
from Note import nn

# Create an instance of the voting_attention layer
voting_att = nn.voting_attention(n_head=8, head_size=64)

# Generate some sample data
encoder_outputs = tf.random.normal((2, 5, 10, 512))
doc_attention_mask = tf.ones((2, 5))

# Apply voting attention
output = voting_att(encoder_outputs, doc_attention_mask)
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

# image preprocessing

## center_crop

The `center_crop` layer crops the center of the input image to the specified height and width. If the input image is smaller than the target size, it will be resized instead.

**Initialization Parameters**

- **`height`** (int): The target height of the cropped image.
- **`width`** (int): The target width of the cropped image.
- **`dtype`** (str, optional): The data type for computations. Default is `'float32'`.

**Methods**

- **`__call__(self, data)`**: Crops or resizes the input `data` to the specified dimensions.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

crop_layer = nn.center_crop(height=128, width=128)
input_image = tf.random.normal((2, 64, 64, 3))
output = crop_layer(input_image)
```

## random_brightness

The `random_brightness` layer adjusts the brightness of the input image by a random factor within a specified range.

**Initialization Parameters**

- **`factor`** (float or tuple): The range of brightness adjustment.
- **`value_range`** (tuple, optional): The value range of the input data. Default is `(0, 255)`.
- **`seed`** (int, optional): Seed for random number generation. Default is `7`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Adjusts the brightness of the input `data`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn
brightness_layer = nn.random_brightness(factor=0.2)
input_image = tf.random.normal((2, 64, 64, 3))
output = brightness_layer(input_image)
```

## random_crop

The `random_crop` layer randomly crops the input image to the specified height and width during training.

**Initialization Parameters**

- **`height`** (int): The target height of the cropped image.
- **`width`** (int): The target width of the cropped image.
- **`seed`** (int, optional): Seed for random number generation. Default is `7`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Randomly crops the input `data`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

crop_layer = nn.random_crop(height=128, width=128)
input_image = tf.random.normal((2, 64, 64, 3))
output = crop_layer(input_image)
```

## random_height

The `random_height` layer randomly varies the height of the input image during training by a specified factor.

**Initialization Parameters**

- **`factor`** (float or tuple): The range of height adjustment.
- **`interpolation`** (str, optional): Interpolation method for resizing. Default is `"bilinear"`.
- **`seed`** (int, optional): Seed for random number generation. Default is `7`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Randomly adjusts the height of the input `data`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

height_layer = nn.random_height(factor=0.2)
input_image = tf.random.normal((2, 64, 64, 3))
output = height_layer(input_image)
```

## random_rotation

The `random_rotation` layer randomly rotates the input image during training by a specified angle.

**Initialization Parameters**

- **`factor`** (float or tuple): The range of rotation angles as a fraction of 2π.
- **`fill_mode`** (str, optional): How to fill points outside the boundaries. Default is `"reflect"`.
- **`interpolation`** (str, optional): Interpolation method. Default is `"bilinear"`.
- **`seed`** (int, optional): Seed for random number generation. Default is `7`.
- **`fill_value`** (float, optional): Fill value for points outside the boundaries when `fill_mode` is `"constant"`. Default is `0.0`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Randomly rotates the input `data`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

rotation_layer = nn.random_rotation(factor=0.2)
input_image = tf.random.normal((2, 64, 64, 3))
output = rotation_layer(input_image)
```

## random_translation

The `random_translation` layer randomly translates the input image during training by a specified factor.

**Initialization Parameters**

- **`height_factor`** (float or tuple): The range of vertical translation.
- **`width_factor`** (float or tuple): The range of horizontal translation.
- **`fill_mode`** (str, optional): How to fill points outside the boundaries. Default is `"reflect"`.
- **`interpolation`** (str, optional): Interpolation method. Default is `"bilinear"`.
- **`seed`** (int, optional): Seed for random number generation. Default is `7`.
- **`fill_value`** (float, optional): Fill value for points outside the boundaries when `fill_mode` is `"constant"`. Default is `0.0`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Randomly translates the input `data`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

translation_layer = nn.random_translation(height_factor=0.2, width_factor=0.2)
output = translation_layer(input_image)
```

Here's the documentation for the preprocessing layers following the provided format:

## random_width

The `random_width` class implements a preprocessing layer that randomly varies the width of an image during training.

**Initialization Parameters**

- **`factor`** (float or tuple): A positive float (fraction of original width) or a tuple of two floats representing the lower and upper bounds for resizing horizontally. For example, `factor=(0.2, 0.3)` results in a width change by a random amount in the range `[20%, 30%]`.
- **`interpolation`** (str): The interpolation method. Options are `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`. Default is `"bilinear"`.
- **`seed`** (int): Seed for random number generation.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Applies random width adjustment to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`train_flag`** (bool, optional): Specifies whether the layer is in training mode. Default is `True`.

  - **Returns**: Width-adjusted output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the random width layer
rw = nn.random_width(factor=(0.2, 0.3))

# Generate some sample data
data = tf.random.normal((2, 64, 64, 3))

# Apply random width adjustment
output = rw(data)
```

## random_zoom

The `random_zoom` class implements a preprocessing layer that randomly zooms images during training.

**Initialization Parameters**

- **`height_factor`** (float or tuple): A float represented as a fraction of the value, or a tuple of two floats representing the lower and upper bounds for zooming vertically. A positive value means zooming out, while a negative value means zooming in.
- **`width_factor`** (float or tuple, optional): A float or a tuple of two floats representing the lower and upper bounds for zooming horizontally. Defaults to `None`.
- **`fill_mode`** (str): Mode for filling points outside the boundaries of the input. Options are `"constant"`, `"reflect"`, `"wrap"`, `"nearest"`. Default is `"reflect"`.
- **`interpolation`** (str): Interpolation mode. Options are `"nearest"`, `"bilinear"`. Default is `"bilinear"`.
- **`seed`** (int): Seed for random number generation.
- **`fill_value`** (float): Value to be filled outside the boundaries when `fill_mode="constant"`. Default is `0.0`.

**Methods**

- **`__call__(self, data, train_flag=True)`**: Applies random zooming to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.
    - **`train_flag`** (bool, optional): Specifies whether the layer is in training mode. Default is `True`.

  - **Returns**: Zoom-adjusted output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the random zoom layer
rz = nn.random_zoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3))

# Generate some sample data
data = tf.random.normal((2, 64, 64, 3))

# Apply random zoom adjustment
output = rz(data)
```

## rescaling

The `rescaling` class implements a preprocessing layer that rescales input values to a new range.

**Initialization Parameters**

- **`scale`** (float): The scale to apply to the inputs.
- **`offset`** (float, optional): The offset to apply to the inputs. Default is `0.0`.

**Methods**

- **`__call__(self, data)`**: Applies rescaling to the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: Rescaled output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the rescaling layer
rs = nn.rescaling(scale=1./255)

# Generate some sample data
data = tf.random.uniform((2, 64, 64, 3), maxval=255, dtype=tf.float32)

# Apply rescaling
output = rs(data)
```

## resizing

The `resizing` class implements a preprocessing layer that resizes images.

**Initialization Parameters**

- **`height`** (int): The height of the output shape.
- **`width`** (int): The width of the output shape.
- **`interpolation`** (str): The interpolation method. Options are `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`. Default is `"bilinear"`.
- **`crop_to_aspect_ratio`** (bool, optional): If `True`, resizes the images without aspect ratio distortion. Default is `False`.

**Methods**

- **`__call__(self, data)`**: Resizes the input `data`.

  - **Parameters**:
    - **`data`**: Input tensor.

  - **Returns**: Resized output tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create an instance of the resizing layer
resize = nn.resizing(height=128, width=128)

# Generate some sample data
data = tf.random.normal((2, 64, 64, 3))

# Apply resizing
output = resize(data)
```

## transform

Applies projective transformations to a batch of images.

- **Parameters:**
  - **`images`**: Tensor of shape `(num_images, num_rows, num_columns, num_channels)`.
  - **`transforms`**: Transformation matrices, either a vector of length 8 or tensor of size `N x 8`.
  - **`fill_mode`**: Mode to fill points outside boundaries (`"constant"`, `"reflect"`, `"wrap"`, `"nearest"`). Default is `"reflect"`.
  - **`fill_value`**: Fill value for `"constant"` mode. Default is `0.0`.
  - **`interpolation`**: Interpolation mode (`"nearest"`, `"bilinear"`). Default is `"bilinear"`.
  - **`output_shape`**: Output shape `[height, width]`. If `None`, output is same size as input.

- **Returns:** Transformed image tensor.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

images = tf.random.normal([5, 256, 256, 3])
transforms = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], shape=(1, 8), dtype=tf.float32)
transformed_images = nn.transform(images, transforms)
```
