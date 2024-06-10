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

# attention

This class implements an attention mechanism for neural networks, supporting both dot-product and concatenation-based attention scoring methods. It also allows for optional scaling of attention scores.

**Initialization Parameters**

- `use_scale` (bool): If `True`, scales the attention scores. Default is `False`.
- `score_mode` (str): The method to calculate attention scores. Options are `"dot"` (default) and `"concat"`.
- `dtype` (str): The data type for computations. Default is `'float32'`.

**Methods**

**`__init__(self, use_scale=False, score_mode="dot", dtype='float32')`**

Initializes the attention mechanism with the specified parameters.

- `use_scale` (bool): Whether to scale the attention scores.
- `score_mode` (str): The scoring method to use ("dot" or "concat").
- `dtype` (str): The data type for computations.

**`__call__(self, query, value, key=None)`**

Applies the attention mechanism to the provided tensors.

- `query` (Tensor): The query tensor.
- `value` (Tensor): The value tensor.
- `key` (Tensor, optional): The key tensor. If not provided, `value` is used as the key.

Returns:
- `Tensor`: The result of the attention mechanism applied to the input tensors.

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
- **`beta_initializer`** (str): Initializer for the beta weight. Default is `'zeros'`.
- **`gamma_initializer`** (str): Initializer for the gamma weight. Default is `'ones'`.
- **`moving_mean_initializer`** (str): Initializer for the moving mean. Default is `'zeros'`.
- **`moving_variance_initializer`** (str): Initializer for the moving variance. Default is `'ones'`.
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

# conv1d

The `conv1d` class implements a 1D convolutional layer, which is commonly used in processing sequential data such as time series or audio.

**Initialization Parameters**

- **`filters`** (int): Number of output filters in the convolution.
- **`kernel_size`** (int or list of int): Size of the convolutional kernel.
- **`input_size`** (int, optional): Size of the input channels.
- **`strides`** (int or list of int): Stride size for the convolution. Default is `[1]`.
- **`padding`** (str or list of int): Padding type or size. Default is `'VALID'`.
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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

# dense

The `dense` class implements a fully connected layer, which is a core component of many neural networks. This layer is used to perform a linear transformation on the input data, optionally followed by an activation function.

**Initialization Parameters**

- **`output_size`** (int): Number of output units (neurons) in the dense layer.
- **`input_size`** (int, optional): Number of input units (neurons) in the dense layer. If not provided, it will be inferred from the input data.
- **`weight_initializer`** (str): Initializer for the weight matrix. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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
- **`weight_initializer`** (str): Initializer for the weight tensor. Default is `'Xavier'`.
- **`bias_initializer`** (str): Initializer for the bias vector. Default is `'zeros'`.
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

# group_norm

The `group_norm` class implements Group Normalization, a technique that divides channels into groups and normalizes each group independently. This can be more stable than batch normalization for small batch sizes.

**Initialization Parameters**

- **`groups`** (int, default=32): Number of groups for normalization. Must be a divisor of the number of channels.
- **`input_size`** (int, optional): Size of the input dimension. If not provided, it will be inferred from the input data.
- **`axis`** (int or list/tuple, default=-1): Axis or axes to normalize across. Typically the feature axis.
- **`epsilon`** (float, default=1e-3): Small constant to avoid division by zero.
- **`center`** (bool, default=True): If `True`, add offset `beta`.
- **`scale`** (bool, default=True): If `True`, multiply by `gamma`.
- **`beta_initializer`** (str, default="zeros"): Initializer for the beta parameter.
- **`gamma_initializer`** (str, default="ones"): Initializer for the gamma parameter.
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
- **`weight_initializer`** (str, default='Xavier'): Initializer for the weight matrices.
- **`bias_initializer`** (str, default='zeros'): Initializer for the bias vectors.
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
- **`weight_initializer`** (str, default='Xavier'): Initializer for the weight matrix.
- **`bias_initializer`** (str, default='zeros'): Initializer for the bias vector.
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
- **`weight_initializer`** (str, default='Xavier'): The method to initialize weights.
- **`bias_initializer`** (str, default='zeros'): The method to initialize biases.
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
- **`weight_initializer`** (str, default='Xavier'): Method for weight initialization.
- **`bias_initializer`** (str, default='zeros'): Method for bias initialization.
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
- **`beta_initializer`** (str, default='zeros'): Initializer for the beta parameter.
- **`gamma_initializer`** (str, default='ones'): Initializer for the gamma parameter.
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
- **`weight_initializer`** (str, default='Xavier'): Initializer for the weights.
- **`bias_initializer`** (str, default='zeros'): Initializer for the biases.
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
- **`weight_initializer`** (str, default='Xavier'): Initializer for the weights.
- **`bias_initializer`** (str, default='zeros'): Initializer for the biases.
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
- **`weight_initializer`** (str, default='Xavier'): Initializer for the weights.
- **`bias_initializer`** (str, default='zeros'): Initializer for the biases.
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
- **`weight_initializer`** (str, default='Xavier'): Initializer for the weights.
- **`bias_initializer`** (str, default='zeros'): Initializer for the biases.
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
