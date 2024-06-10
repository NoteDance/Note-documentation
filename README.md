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
This module implements a long short-term memory (LSTM) layer, which can process the input data in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the LSTM class, and specify the output size, the input size, the weight initializer, the bias initializer, the data type, the return sequence flag, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformations or not. The activation functions should be callable objects that take a tensor as input and return a tensor as output.
- Second, pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- Last, return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the LSTM output after applying the input gate, forget gate, output gate and cell state update to the input data and the previous hidden state and cell state.

For example:

```python
# Create an LSTM layer with 32 hidden units
lstm_layer = LSTM(32, 16)
# Apply the LSTM layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = lstm_layer(data=input_data)
# The output_data will have a shape of [64, 32]
```

# LSTMCell
This module implements a long short-term memory (LSTM) cell, which can process the input data and the previous state in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the LSTMCell class, and specify the weight shape, the weight initializer, the bias initializer, the data type, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The activation functions should be callable objects that take a tensor as input and return a tensor as output. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, pass the input data and the previous state as the data and state arguments. The input data should be a tensor of shape [batch_size, input_size], where batch_size is the number of samples in a batch and input_size is the dimension of the input features. The previous state should be a tensor of shape [batch_size,
  hidden_size], where hidden_size is the dimension of the hidden state.
- Last, return a tuple of two tensors: output and new_state. The output is a tensor of shape [batch_size,
  hidden_size], which is the output of the cell at the current time step. The new_state is a tensor of shape [batch_size,
  hidden_size], which is the updated hidden state for the next time step.

For example:

```python
# Create an LSTM cell with 32 hidden units
lstm_cell = LSTMCell(weight_shape=[16, 32])
# Apply the LSTM cell to a batch of input data of shape [64, 16] and a previous state of shape [64, 32]
input_data = tf.random.normal([64, 16])
prev_state = tf.random.normal([64, 32])
output_data, new_state = lstm_cell(data=input_data, state=prev_state)
# The output_data will have a shape of [64, 32]
# The new_state will have a shape of [64, 32]
```

# Linformer_self_attention
This module implements the Linformer self-attention mechanism, which is a fast and scalable way to compute attention using positive orthogonal random features. The usage of this module is as follows:

- First, create an instance of the Linformer_self_attention class, and specify the dimension, the sequence length, and other optional parameters such as k, heads, dim_head, one_kv_head, share_kv, dropout, and dtype.
- Second, pass the input tensor as the x argument. You can also pass a different tensor as the context argument if you want to use cross-attention. The output method will return a tensor of shape [batch_size, sequence_length, dimension], which is the Linformer self-attention output.

For example:

```python
# Create a Linformer self-attention layer with 64 dimension, 128 sequence length, 32 k, 8 heads, and 0.1 dropout
linformer = Linformer_self_attention(dim=64, seq_len=128, k=32, heads=8, dropout=0.1)
# Apply the Linformer self-attention layer to a batch of input embeddings of shape [16, 128, 64]
input_embeddings = tf.random.normal([16, 128, 64])
output_embeddings = linformer(input_embeddings)
# The output_embeddings will have a shape of [16, 128, 64]
```

# layer_norm
This module implements a layer normalization layer, which is a common technique for deep learning models. Layer normalization can normalize each neuron of each sample, making its mean 0 and variance 1. This can avoid the internal covariate shift, accelerate the model convergence, and improve the model generalization ability. This method was proposed by Ba et al. in 2016.

The usage of this module is as follows:

- First, create an instance of the layer_normalization class, and specify the axis or axes to normalize, the epsilon, and other optional parameters such as input size, center, scale, beta initializer, gamma initializer, and dtype.
- Second, pass the input tensor as the data argument. The output method will apply layer normalization to the input tensor and return a normalized tensor of the same shape.

For example:

```python
# Create a layer normalization layer with axis -1
ln = layer_normalization(128, axis=-1)
# Apply the layer normalization layer to a batch of input data of shape [64, 128]
input_data = tf.random.normal([64, 128])
output_data = ln(input_data)
# The output_data will have a shape of [64, 128]
```

# multihead_attention
This module defines a multihead_attention class that implements a multi-head attention layer. A multi-head attention layer is a sublayer of the standard transformer layer that can learn the relevance and dependency of different tokens in a sequence. The usage of this module is as follows:

- First, create an instance of the multihead_attention class, and specify the n_state, n_head, and other optional parameters such as weight_initializer, bias_initializer, dtype, and use_bias. The n_state is the dimensionality of the query, key, and value tensors after the linear transformation. The n_head is the number of attention heads. The use_bias indicates whether to use a bias term after the linear transformations or not.
- Second, pass the input data as the x argument. Optionally, you can also pass another input data as the xa argument, which will be used as the key and value for the attention computation. If xa is not provided, x will be used as the query, key, and value. You can also pass a mask argument to mask out some tokens from the attention computation. The input data should be a tensor of shape [batch_size, seq_length, n_state], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and n_state is the dimension of the input features at each time step. The mask should be a tensor of shape [batch_size, seq_length_q, seq_length_k], where seq_length_q is the number of time steps in x and seq_length_k is the number of time steps in xa (or x if xa is not provided).
- Last, return a tuple of two tensors: output_data and qk. The output_data is a tensor of shape [batch_size, seq_length_q,
  n_state], which is the multi-head attention output. The qk is a tensor of shape [batch_size, n_head, seq_length_q,
  seq_length_k], which is the scaled dot product attention score for each head. The output_data is computed by applying query, key, and value projections to the input data, followed by scaled dot product attention with optional masking, concatenation of the attention outputs from each head, and output projection. The qk is computed by applying query and key projections to the input data, followed by scaled dot product attention with optional masking.

For example:

```python
# Create a multi-head attention layer with 64 n_state and 8 n_head
multihead_attention_layer = multihead_attention(8, 64)
# Apply the multi-head attention layer to a batch of input data of shape [32, 20, 64]
input_data = tf.random.normal([32, 20, 64])
output_data, qk = multihead_attention_layer(x=input_data)
# The output_data will have a shape of [32, 20, 64]
# The qk will have a shape of [32, 8, 20, 20]
```

# RNN
This module implements a simple recurrent neural network (RNN) layer, which can process the input data in a sequential manner and learn short-term dependencies. The usage of this module is as follows:

- First, create an instance of the RNN class, and specify the output size, the input size, the weight initializer, the bias initializer, the activation function, the data type, the return sequence flag, and the use bias flag. The weight shape should be a list of two integers: [input_size, hidden_size]. The activation function should be a string that matches one of the keys in the activation_dict dictionary. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformation or not.
- Second, pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- Last, return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the RNN output after applying the activation function to the linear combination of the input data and the previous hidden state.

For example:

```python
# Create an RNN layer with 32 hidden units and tanh activation
rnn_layer = RNN(32, 16, activation='tanh')
# Apply the RNN layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = rnn_layer(data=input_data)
# The output_data will have a shape of [64, 32]
```

# RNNCell
This module implements a recurrent neural network (RNN) cell, which can process the input data and the previous state in a sequential manner and learn short-term dependencies. The usage of this module is as follows:

- First, create an instance of the RNNCell class, and specify the weight shape, the weight initializer, the bias initializer, the activation function, the use bias flag, the trainable flag, and the data type. The weight shape should be a list of two integers: [input_size, output_size]. The activation function should be a string that matches one of the keys in the activation function dictionary. The use bias flag indicates whether to add a bias term to the linear transformations or not. The trainable flag indicates whether to update the parameters during training or not. The data type should be a string that represents a valid TensorFlow data type.
- Second, pass the input data and the previous state as the data and state arguments. The input data should be a tensor of shape [batch_size, input_size], where batch_size is the number of samples in a batch and input_size is the dimension of the input features. The previous state should be a tensor of shape [batch_size,
  output_size], where output_size is the dimension of the output state.
- Last, return a tensor of shape [batch_size,
  output_size], which is the output of the cell at the current time step.

For example:

```python
# Create an RNN cell with 32 output units and tanh activation function
rnn_cell = RNNCell(weight_shape=[16, 32], activation='tanh')
# Apply the RNN cell to a batch of input data of shape [64, 16] and a previous state of shape [64, 32]
input_data = tf.random.normal([64, 16])
prev_state = tf.random.normal([64, 32])
output_data = rnn_cell(data=input_data,state=prev_state)
# The output_data will have a shape of [64, 32]
```

# Transformer
This class implements a Transformer model, which is a type of neural network that can handle sequential data, such as natural language or speech. The usage of this class is as follows:

- First, create an instance of the Transformer class, and specify the dimension of the input and output vectors, the number of attention heads, the number of encoder and decoder layers, and other optional parameters such as feedforward dimension, dropout probability, activation function, custom encoder and decoder submodules, layer normalization epsilon, norm first flag, bias flag, device, and data type.
- Second, pass the source and target sequences as the src and tgt arguments. You can also specify optional masks for the source, target, and memory sequences.
- Last, return a tensor of shape [batch_size, tgt_len, d_model], which is the output sequence generated by the Transformer model.

For example:

```python
# Create a Transformer model with 512 input and output dimension, 8 attention heads,
# 6 encoder and decoder layers, ReLU activation
transformer = Transformer(d_model=512, nhead=8, num_encoder_layers=6,
                          num_decoder_layers=6, activation='relu')
# Apply the Transformer model to a batch of source and target data of shape [10, 50, 512]
src = tf.random.normal([10, 50, 512])
tgt = tf.random.normal([10, 50, 512])
output = transformer(src, tgt)
# The output will have a shape of [10, 50, 512]
```

# talking_heads_attention
This module implements a talking heads attention mechanism, which can enhance the expressiveness and diversity of self-attention and multi-head attention. The usage of this module is as follows:

- First, create an instance of the talking_heads_attention class, and specify the qkv_rank, attention_axes, dropout_rate, initializer, dtype, and other optional parameters. The qkv_rank is the rank of the query, key, and value tensors. The attention_axes is a tuple of axes over which the attention is computed. The dropout_rate is the probability of dropping out attention scores. The initializer is the name of the weight initializer. The dtype is the data type of the tensors.
- Second, pass the query_tensor, key_tensor, value_tensor, and attention_mask as arguments. You can also specify a different train_flag to indicate whether to apply dropout or not. The query_tensor, key_tensor, and value_tensor are tensors of shape [batch_size, ..., num_heads, depth]. The attention_mask is a tensor of shape [batch_size, ..., 1, 1] or [batch_size, ..., 1, key_length], where key_length is the size of the last dimension of key_tensor.
- Last, return a tuple of (attention_output, attention_scores). The attention_output is a tensor of shape [batch_size, ..., num_heads, depth], which is the result of applying attention to value_tensor. The attention_scores is a tensor of shape [batch_size, num_heads, ..., ...], which is the normalized and projected attention scores.

For example:

```python
# Create a talking heads attention layer with qkv_rank=4, attention_axes=(1,), dropout_rate=0.1
tha = talking_heads_attention(qkv_rank=4, attention_axes=(1,), dropout_rate=0.1)
# Apply the talking heads attention layer to a batch of query, key, and value tensors of shape [32, 64, 8, 128]
query_tensor = tf.random.normal([32, 64, 8, 128])
key_tensor = tf.random.normal([32, 64, 8, 128])
value_tensor = tf.random.normal([32, 64, 8, 128])
# The attention_mask is None by default
attention_output, attention_scores = tha(query_tensor,
                                                key_tensor,
                                                value_tensor)
# The attention_output will have a shape of [32, 64, 8, 128]
# The attention_scores will have a shape of [32, 8, 64, 64]
```

# separable_conv1d
This module implements a separable convolutional layer, which can apply a depthwise convolution and a pointwise convolution to an input tensor and produce a feature map. The usage of this module is as follows:

- First, create an instance of the separable_conv1d class, and specify the number of output filters, the kernel size, the depth multiplier, and other optional parameters such as input size, strides, padding mode, data format, dilation rate, weight initializer, bias initializer, activation function, use bias flag, and data type.
- Second, pass the input tensor as the data argument. The input tensor should be a three-dimensional tensor of shape [batch_size, length, channels], where batch_size is the number of samples in a batch, length is the dimension of the input sequence, and channels is the dimension of the input features.
- Last, return a tensor of shape [batch_size, new_length, filters], where new_length is the dimension of the output sequence after applying the convolution operation, and filters is the number of output filters.

For example:

```python
# Create a separable convolution layer with 64 output filters, 5 kernel size, 2 depth multiplier
separable_conv1d = separable_conv1d(filters=64, kernel_size=5, depth_multiplier=2,input_size=16)
# Apply the separable convolution layer to a batch of input data of shape [32, 100, 16]
input_data = tf.random.normal([32, 100, 16])
output_data = separable_conv1d(input_data)
# The output_data will have a shape of [32, 96, 64]
```

# separable_conv2d
This module implements a separable convolutional layer, which can apply a depthwise convolution and a pointwise convolution to an input tensor and produce a feature map. The usage of this module is as follows:

- First, create an instance of the separable_conv2d class, and specify the number of output filters, the kernel size, the depth multiplier, and other optional parameters such as input size, strides, padding mode, data format, dilation rate, weight initializer, bias initializer, activation function, use bias flag, and data type.
- Second, pass the input tensor as the data argument. The input tensor should be a four-dimensional tensor of shape [batch_size, height, width, channels], where batch_size is the number of samples in a batch, height and width are the dimensions of the input image, and channels is the dimension of the input features.
- Last, return a tensor of shape [batch_size, new_height, new_width, filters], where new_height and new_width are the dimensions of the output image after applying the convolution operation, and filters is the number of output filters.

For example:

```python
# Create a separable convolution layer with 64 output filters, 5x5 kernel size, 2 depth multiplier
separable_conv2d = separable_conv2d(filters=64, kernel_size=[5, 5], depth_multiplier=2,input_size=3)
# Apply the separable convolution layer to a batch of input data of shape [32, 28, 28, 3]
input_data = tf.random.normal([32, 28, 28, 3])
output_data = separable_conv2d(input_data)
# The output_data will have a shape of [32, 24, 24, 64]
```

# stochastic_depth
This module implements a stochastic depth layer, which can randomly skip some layers of a neural network and scale the remaining layers by a factor of 1/(1-drop_path_rate). The usage of this module is as follows:

- First, create an instance of the stochastic_depth class, and specify the drop path rate, and other optional parameters such as noise shape, seed, and data type.
- Second, pass the input tensor as the data argument. You can also specify a boolean flag to indicate whether the stochastic depth layer is in training mode or inference mode.
- Last, return a tensor of the same shape as the input tensor, which is the stochastic depth output.

For example:

```python
# Create a stochastic depth layer with 0.2 drop path rate
stochastic_depth = stochastic_depth(drop_path_rate=0.2)
# Apply the stochastic depth layer to a batch of input data of shape [32, 100]
input_data = tf.random.normal([32, 100])
output_data = stochastic_depth(input_data)
# The output_data will have a shape of [32, 100], and some vectors will be zeroed out
```

# zeropadding1d
This module implements a 1D zero-padding layer, which can add zeros at the beginning and end of a 1D input tensor (e.g. temporal sequence). The usage of this module is as follows:

- First, create an instance of the zeropadding1d class, and optionally specify the input size and the padding size. If the input size is given, the output size will be the same as the input size. If the padding size is given, it can be either an integer or a tuple of two integers. If it is an integer, it will use the same padding for both sides. If it is a tuple of two integers, it will use the first integer for the left padding and the second integer for the right padding.
- Second, pass the input tensor as the data argument. You can also specify a different padding size for this method, which will override the padding size given in the constructor.
- Last, return a tensor of shape [batch_size, length + padding, channels], which is the 1D zero-padded output.

For example:

```python
# Create a 1D zero-padding layer with padding size 2
zeropadding1d = zeropadding1d(padding=2)
# Apply the 1D zero-padding layer to a batch of input data of shape [32, 96, 100]
input_data = tf.random.normal([32, 96, 100])
output_data = zeropadding1d(input_data)
# The output_data will have a shape of [32, 100, 100]
```

# zeropadding2d
This module implements a 2D zero-padding layer, which can add rows and columns of zeros at the top, bottom, left and right side of a 2D input tensor (e.g. image). The usage of this module is as follows:

- First, create an instance of the zeropadding2d class, and optionally specify the input size and the padding size. If the input size is given, the output size will be the same as the input size. If the padding size is given, it can be either an integer or a tuple of two tuples. If it is an integer, it will use the same padding for all sides. If it is a tuple of two tuples, it will use the first tuple for the height padding and the second tuple for the width padding. Each tuple should have two elements, representing the padding before and after the dimension.
- Second, pass the input tensor as the data argument. You can also specify a different padding size for this method, which will override the padding size given in the constructor.
- Last, return a tensor of shape [batch_size, height, width, channels], which is the 2D zero-padded output.

For example:

```python
# Create a 2D zero-padding layer with padding size 2
zeropadding2d = zeropadding2d(padding=2)
# Apply the 2D zero-padding layer to a batch of input data of shape [32, 96, 100, 3]
input_data = tf.random.normal([32, 96, 100, 3])
output_data = zeropadding2d(input_data)
# The output_data will have a shape of [32, 100, 104, 3]
```

# zeropadding3d
This module implements a 3D zero-padding layer, which can add rows, columns and slices of zeros at the top, bottom, left, right, front and back side of a 3D input tensor (e.g. volumetric image). The usage of this module is as follows:

- First, create an instance of the zeropadding3d class, and optionally specify the input size and the padding size. If the input size is given, the output size will be the same as the input size. If the padding size is given, it can be either an integer or a tuple of three tuples. If it is an integer, it will use the same padding for all sides. If it is a tuple of three tuples, it will use the first tuple for the depth padding, the second tuple for the height padding and the third tuple for the width padding. Each tuple should have two elements, representing the padding before and after the dimension.
- Second, pass the input tensor as the data argument. You can also specify a different padding size for this method, which will override the padding size given in the constructor.
- Last, return a tensor of shape [batch_size, depth + padding, height + padding, width + padding, channels], which is the 3D zero-padded output.

For example:

```python
# Create a 3D zero-padding layer with padding size 2
zeropadding3d = zeropadding3d(padding=2)
# Apply the 3D zero-padding layer to a batch of input data of shape [32, 96, 100, 100, 3]
input_data = tf.random.normal([32, 96, 100, 100, 3])
output_data = zeropadding3d(input_data)
# The output_data will have a shape of [32, 100, 104, 104, 3]
```
