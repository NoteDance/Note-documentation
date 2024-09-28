# assign_param

The `assign_param` function copies values from one set of parameters to another. This is useful in scenarios where you need to update or synchronize model parameters.

- **Parameters**
  - `param1`: The target parameters to which values will be assigned.
  - `param2`: The source parameters from which values will be copied.

- **Returns**: None.

- **Example:**
  ```python
  # Example parameters (e.g., model weights)
  model.param = [...]  # Target parameters
  param = [...]  # Source parameters
  
  from Note import nn
  nn.assign_param(model.param, param)
  ```

This function leverages TensorFlow's `state_ops.assign` for assignment operations and `nest.flatten` to handle nested structures of parameters.

# conv2d_func

The `conv2d_func` function performs a 2D convolution operation with support for various configurations including groups, padding, and dilations.

- **Parameters**
  - `input`: Input tensor.
  - `weight`: Convolution filter tensor.
  - `bias` (optional): Bias tensor. Default is `None`.
  - `strides`: Convolution strides. Default is `1`.
  - `padding`: Padding value or type ('SAME' or 'VALID'). Default is `0`.
  - `dilations`: Dilation rate. Default is `1`.
  - `groups`: Number of groups for grouped convolution. Default is `1`.

- **Returns:** Output tensor after applying the convolution operation.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn
  
  # Define input and filter tensors
  input = tf.random.normal((1, 64, 64, 3))
  weight = tf.random.normal((3, 3, 3, 16))
  
  # Apply conv2d_func
  output = nn.conv2d_func(input, weight)
  ```

# create_aa

The `create_aa` function creates an anti-aliasing layer for convolutional neural networks, which helps to reduce aliasing artifacts during downsampling operations.

- **Parameters**
  - `aa_layer`: Type of anti-aliasing layer (e.g., 'avg', 'blur').
  - `channels` (optional): Number of channels in the input tensor.
  - `stride`: Stride value for the anti-aliasing operation. Default is `2`.
  - `enable`: Boolean flag to enable or disable the anti-aliasing layer. Default is `True`.
  - `noop`: Function to use if anti-aliasing is disabled. Default is `identity`.

- **Returns:** Anti-aliasing layer or the `noop` function if anti-aliasing is disabled.

- **Example:**
  ```python
  from Note import nn
  
  # Create an anti-aliasing layer
  aa_layer = nn.create_aa('avg', channels=16, stride=2)
  
  # Apply anti-aliasing layer
  output = aa_layer(input)
  ```

# cosine_similarity

Computes the cosine similarity between two tensors.

- **Parameters:**
  - `x1`, `x2`: Input tensors.
  - `axis`: Axis to compute similarity. Default is `1`.
  - `eps`: Small value to avoid division by zero. Default is `1e-8`.

- **Returns:** Cosine similarity tensor.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn

  x1 = tf.random.normal([10, 128])
  x2 = tf.random.normal([10, 128])
  similarity = nn.cosine_similarity(x1, x2)
  ```

# create_additive_causal_mask

Creates a causal mask for sequence operations.

- **Parameters:**
  - `N`: Size of the sequence.
  - `dtype`: Data type of the mask. Default is `tf.float32`.

- **Returns:** Causal mask tensor.

- **Example:**
  ```python
  from Note import nn
  
  mask = nn.create_additive_causal_mask(10)
  ```

# gather_mm

Gathers data according to given indices and performs matrix multiplication.

- **Parameters:**
  - `a`: 3-D tensor of shape `(N, M, D1)` or 2-D tensor of shape `(N, D1)`.
  - `b`: 3-D tensor of shape `(R, D1, D2)`.
  - `idx_b`: 1-D integer tensor of shape `(N,)`.

- **Returns:** Dense matrix of shape `(N, M, D2)` or `(N, D2)`.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn

  a = tf.random.normal([5, 10, 20])
  b = tf.random.normal([15, 20, 25])
  idx_b = tf.constant([0, 1, 2, 3, 4])
  result = nn.gather_mm(a, b, idx_b)
  ```

# interpolate

Performs interpolation on a tensor.

- **Parameters:**
  - `input`: Input tensor.
  - `size`: Output size `[height, width]`.
  - `scale_factor`: Scale factor for resizing.
  - `recompute_scale_factor`: Whether to recompute scale factor. Default is `False`.
  - `mode`: Interpolation mode (`"nearest"`, `"bilinear"`, `"bicubic"`). Default is `"nearest"`.
  - `align_corners`: If `True`, aligns corners of input and output. Default is `False`.
  - `antialias`: Whether to use an anti-aliasing filter when downsampling an image.

- **Returns:** Interpolated tensor.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn

  input = tf.random.normal([5, 32, 32, 3])
  resized = nn.interpolate(input, size=[64, 64], mode='bilinear')
  ```

# pairwise_distance

Calculates pairwise distance between two tensors.

- **Parameters:**
  - `x`, `y`: Input tensors.
  - `p`: Norm degree. Default is `2`.
  - `eps`: Small value to avoid numerical issues. Default is `1e-6`.
  - `keepdim`: Whether to keep dimensions. Default is `False`.

- **Returns:** Pairwise distance tensor.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn

  x = tf.random.normal([10, 128])
  y = tf.random.normal([10, 128])
  distance = nn.pairwise_distance(x, y)
  ```

# resample_abs_pos_embed

The `resample_abs_pos_embed` function resamples absolute position embeddings to a new size, which is useful when the input resolution to a model changes.

- **Parameters**
  - `posemb` (Tensor): The input position embedding tensor of shape (B, N, C).
  - `new_size` (List[int]): The desired new size (height, width) for the position embeddings.
  - `old_size` (Optional[List[int]]): The original size (height, width) of the position embeddings. If not provided, it assumes the position embeddings are square.
  - `num_prefix_tokens` (int): Number of prefix tokens (e.g., class token). Default is `1`.
  - `interpolation` (str): Interpolation method to use. Default is `'bicubic'`.
  - `antialias` (bool): Whether to apply antialiasing when resizing. Default is `True`.
  - `verbose` (bool): If `True`, logs information about the resizing process. Default is `False`.

- **Returns:** The resampled position embedding tensor.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn
  import math
  
  # Create a sample position embedding tensor
  posemb = tf.random.normal((1, 197, 768))
  
  # Define new size
  new_size = [16, 16]
  
  # Resample position embeddings
  resampled_posemb = nn.resample_abs_pos_embed(posemb, new_size)
  ```

# resample_abs_pos_embed_nhwc

The `resample_abs_pos_embed_nhwc` function resamples absolute position embeddings for tensors in NHWC format (height, width, channels).

- **Parameters:**
  - `posemb` (Tensor): The input position embedding tensor in NHWC format.
  - `new_size` (List[int]): The desired new size (height, width) for the position embeddings.
  - `interpolation` (str): Interpolation method to use. Default is `'bicubic'`.
  - `antialias` (bool): Whether to apply antialiasing when resizing. Default is `True`.
  - `verbose` (bool): If `True`, logs information about the resizing process. Default is `False`.

- **Returns:** The resampled position embedding tensor in NHWC format.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn
  
  # Create a sample position embedding tensor in NHWC format
  posemb_nhwc = tf.random.normal((1, 14, 14, 768))
  
  # Define new size
  new_size = [16, 16]
  
  # Resample position embeddings
  resampled_posemb_nhwc = nn.resample_abs_pos_embed_nhwc(posemb_nhwc, new_size)
  ```

# positional_encoding

Generates positional encoding for a sequence.

- **Parameters:**
  - `max_len`: Maximum length of the sequence.
  - `d_model`: Dimensionality of the encoding.

- **Returns:** Positional encoding tensor.

- **Example:**
  ```python
  from Note import nn
  
  encoding = nn.positional_encoding(100, 512)
  ```

# scaled_dot_product_attention

Performs scaled dot-product attention.

- **Parameters:**
  - `query`, `key`, `value`: Input tensors.
  - `attn_mask`: Optional attention mask.
  - `dropout_p`: Dropout probability. Default is `0.0`.
  - `is_causal`: If `True`, applies causal mask.
  - `scale`: Optional scaling factor. Default is `None`.

- **Returns:** Tensor after applying attention.

- **Example:**
  ```python
  import tensorflow as tf
  from Note import nn

  query = tf.random.normal([5, 10, 64])
  key = tf.random.normal([5, 10, 64])
  value = tf.random.normal([5, 10, 64])
  attn_output = nn.scaled_dot_product_attention(query, key, value)
  ```

# trunc_normal_

The `trunc_normal_` function fills a tensor with values drawn from a truncated normal distribution. This distribution is bounded by specified minimum and maximum values, ensuring that all values in the tensor fall within these bounds.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the truncated normal distribution.
- **`mean`** (float, optional): The mean of the normal distribution. Default is `0.`.
- **`std`** (float, optional): The standard deviation of the normal distribution. Default is `1.`.
- **`a`** (float, optional): The minimum cutoff value. Default is `-2.`.
- **`b`** (float, optional): The maximum cutoff value. Default is `2.`.

**Method**

- **`trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.)`**: Fills the input `tensor` with values drawn from a truncated normal distribution.

  - **Parameters**:
    - **`tensor`**: An n-dimensional `tf.Variable`.
    - **`mean`** (float, optional): The mean of the normal distribution.
    - **`std`** (float, optional): The standard deviation of the normal distribution.
    - **`a`** (float, optional): The minimum cutoff value.
    - **`b`** (float, optional): The maximum cutoff value.

  - **Returns**: The input `tensor` filled with values from the truncated normal distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of trunc_normal_
tensor = tf.Variable(tf.zeros((3, 5)), dtype=tf.float32)
nn.trunc_normal_(tensor)
print(tensor)
```

# trunc_normal_tf_

The `trunc_normal_tf_` function fills a tensor with values drawn from a truncated normal distribution, similar to `trunc_normal_`, but it behaves closer to TensorFlow or JAX implementations. This function first samples the normal distribution with mean=0 and std=1, then scales and shifts the result by the specified mean and standard deviation.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the truncated normal distribution.
- **`mean`** (float, optional): The mean of the normal distribution. Default is `0.`.
- **`std`** (float, optional): The standard deviation of the normal distribution. Default is `1.`.
- **`a`** (float, optional): The minimum cutoff value. Default is `-2.`.
- **`b`** (float, optional): The maximum cutoff value. Default is `2.`.

**Method**

- **`trunc_normal_tf_(tensor, mean=0., std=1., a=-2., b=2.)`**: Fills the input `tensor` with values drawn from a truncated normal distribution.

  - **Parameters**:
    - **`tensor`**: An n-dimensional `tf.Variable`.
    - **`mean`** (float, optional): The mean of the normal distribution.
    - **`std`** (float, optional): The standard deviation of the normal distribution.
    - **`a`** (float, optional): The minimum cutoff value.
    - **`b`** (float, optional): The maximum cutoff value.

  - **Returns**: The input `tensor` filled with values from the truncated normal distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of trunc_normal_tf_
tensor = tf.Variable(tf.zeros((3, 5)), dtype=tf.float32)
nn.trunc_normal_tf_(tensor)
print(tensor)
```

# dirac_

The `dirac_` function initializes a tensor with the Dirac delta function, preserving the identity of the inputs in convolutional layers. This is useful for initializing layers where you want to retain as many input channels as possible.

**Parameters**

- **`tensor`**: A {3, 4, 5}-dimensional `tf.Variable` that will be filled with the Dirac delta function.
- **`groups`** (int, optional): The number of groups in the convolutional layer. Default is `1`.

**Method**

- **`dirac_(tensor, groups=1)`**: Fills the input `tensor` with the Dirac delta function.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of dirac_
tensor = tf.Variable(tf.zeros([5, 5, 16, 3]))
nn.dirac_(tensor)
print(tensor)

tensor = tf.Variable(tf.zeros([5, 5, 24, 3]))
nn.dirac_(tensor, groups=3)
print(tensor)
```

This function is particularly useful for convolutional layers in neural networks, where maintaining the identity of the inputs is important for preserving certain properties of the data as it passes through the network. The `groups` parameter allows for dividing the channels into multiple groups, each preserving the identity independently.

# variance_scaling_

The `variance_scaling_` function initializes a tensor with values from a scaled distribution based on the variance of the input tensor. It supports different modes and distributions.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the specified distribution.
- **`scale`** (float, optional): Scaling factor. Default is `1.0`.
- **`mode`** (str, optional): Mode for calculating the scaling factor. Can be `'fan_in'`, `'fan_out'`, or `'fan_avg'`. Default is `'fan_in'`.
- **`distribution`** (str, optional): Distribution to sample from. Can be `'normal'`, `'truncated_normal'`, or `'uniform'`. Default is `'normal'`.

**Method**

- **`variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal')`**: Fills the input `tensor` with values from the specified scaled distribution.

  - **Parameters**:
    - **`tensor`**: An n-dimensional `tf.Variable`.
    - **`scale`** (float, optional): Scaling factor.
    - **`mode`** (str, optional): Mode for calculating the scaling factor.
    - **`distribution`** (str, optional): Distribution to sample from.

  - **Returns**: The input `tensor` filled with values from the scaled distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of variance_scaling_
tensor = tf.Variable(tf.zeros((3, 5)), dtype=tf.float32)
nn.variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal')
print(tensor)
```

# lecun_normal_

The `lecun_normal_` function initializes a tensor with values from a truncated normal distribution, scaled according to the LeCun initialization method.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the LeCun-normal distribution.

**Method**

- **`lecun_normal_(tensor)`**: Fills the input `tensor` with values from the LeCun-normal distribution.

  - **Parameters**:
    - **`tensor`**: An n-dimensional `tf.Variable`.

  - **Returns**: The input `tensor` filled with values from the LeCun-normal distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of lecun_normal_
tensor = tf.Variable(tf.zeros((3, 5)), dtype=tf.float32)
nn.lecun_normal_(tensor)
print(tensor)
```

# calculate_gain

The `calculate_gain` function returns the recommended gain value for a given nonlinearity function, which is used in weight initialization.

**Parameters**

- **`nonlinearity`** (str): The name of the non-linear function (e.g., `'relu'`, `'leaky_relu'`).
- **`param`** (optional): An optional parameter for the non-linear function (e.g., negative slope for leaky ReLU).

**Method**

- **`calculate_gain(nonlinearity, param=None)`**: Returns the recommended gain value for the given nonlinearity function.

  - **Parameters**:
    - **`nonlinearity`** (str): The name of the non-linear function.
    - **`param`** (optional): An optional parameter for the non-linear function.

  - **Returns**: The recommended gain value for the given nonlinearity function.

**Example Usage**

```python
from Note import nn

# Example usage of calculate_gain
gain = nn.calculate_gain('leaky_relu', 0.2)
print(gain)
```

# xavier_uniform_

The `xavier_uniform_` function initializes a tensor with values from a Xavier uniform distribution, which is used for initializing weights in neural networks.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the Xavier uniform distribution.
- **`gain`** (float, optional): An optional scaling factor. Default is `1.0`.
- **`generator`** (optional): A generator for random number generation. Default is `None`.

**Method**

- **`xavier_uniform_(tensor, gain=1.0, generator=None)`**: Fills the input `tensor` with values from a Xavier uniform distribution.

  - **Parameters**:
    - **`tensor`**: An n-dimensional `tf.Variable`.
    - **`gain`** (float, optional): An optional scaling factor.
    - **`generator`** (optional): A generator for random number generation.

  - **Returns**: The input `tensor` filled with values from the Xavier uniform distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of xavier_uniform_
tensor = tf.Variable(tf.zeros([3, 5]))
nn.xavier_uniform_(tensor, gain=nn.calculate_gain('relu'))
print(tensor)
```

# xavier_normal_

The `xavier_normal_` function initializes a tensor with values from a Xavier normal distribution, which is used for initializing weights in neural networks.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the Xavier normal distribution.
- **`gain`** (float, optional): An optional scaling factor. Default is `1.0`.
- **`generator`** (optional): A generator for random number generation. Default is `None`.

**Method**

- **`xavier_normal_(tensor, gain=1.0, generator=None)`**: Fills the input `tensor` with values from a Xavier normal distribution.

  - **Parameters**:
    - **`tensor`**: An n-dimensional `tf.Variable`.
    - **`gain`** (float, optional): An optional scaling factor.
    - **`generator`** (optional): A generator for random number generation.

  - **Returns**: The input `tensor` filled with values from the Xavier normal distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of xavier_normal_
tensor = tf.Variable(tf.zeros([3, 5]))
nn.xavier_normal_(tensor)
print(tensor)
```

# kaiming_uniform_

The `kaiming_uniform_` function initializes a tensor with values from a Kaiming uniform distribution, also known as He initialization. This method is typically used for layers with ReLU or leaky ReLU activations.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the Kaiming uniform distribution.
- **`a`** (float, optional): The negative slope of the rectifier used after this layer (used only with `'leaky_relu'`). Default is `0`.
- **`mode`** (str, optional): Either `'fan_in'` (default) or `'fan_out'`. `'fan_in'` preserves the variance in the forward pass, while `'fan_out'` preserves it in the backward pass.
- **`nonlinearity`** (str, optional): The non-linear function (`'relu'` or `'leaky_relu'`). Default is `'leaky_relu'`.
- **`generator`** (optional): A generator for random number generation. Default is `None`.

**Method**

- **`kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)`**: Fills the input `tensor` with values from a Kaiming uniform distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of kaiming_uniform_
tensor = tf.Variable(tf.zeros([3, 5]))
nn.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
print(tensor)
```

# kaiming_normal_

The `kaiming_normal_` function initializes a tensor with values from a Kaiming normal distribution, also known as He initialization. This method is typically used for layers with ReLU or leaky ReLU activations.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` that will be filled with values from the Kaiming normal distribution.
- **`a`** (float, optional): The negative slope of the rectifier used after this layer (used only with `'leaky_relu'`). Default is `0`.
- **`mode`** (str, optional): Either `'fan_in'` (default) or `'fan_out'`. `'fan_in'` preserves the variance in the forward pass, while `'fan_out'` preserves it in the backward pass.
- **`nonlinearity`** (str, optional): The non-linear function (`'relu'` or `'leaky_relu'`). Default is `'leaky_relu'`.
- **`generator`** (optional): A generator for random number generation. Default is `None`.

**Method**

- **`kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)`**: Fills the input `tensor` with values from a Kaiming normal distribution.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of kaiming_normal_
tensor = tf.Variable(tf.zeros([3, 5]))
nn.kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
print(tensor)
```

# orthogonal_

The `orthogonal_` function initializes a tensor with a (semi) orthogonal matrix, preserving the orthogonality properties during initialization.

**Parameters**

- **`tensor`**: An n-dimensional `tf.Variable` with at least 2 dimensions.
- **`gain`** (float, optional): An optional scaling factor. Default is `1`.
- **`generator`** (optional): A generator for random number generation. Default is `None`.

**Method**

- **`orthogonal_(tensor, gain=1, generator=None)`**: Fills the input `tensor` with a (semi) orthogonal matrix.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of orthogonal_
tensor = tf.Variable(tf.zeros([3, 5]))
nn.orthogonal_(tensor)
print(tensor)
```

# sparse_

The `sparse_` function initializes a 2D tensor as a sparse matrix, with non-zero elements drawn from a normal distribution.

**Parameters**

- **`tensor`**: A 2-dimensional `tf.Variable`.
- **`sparsity`** (float): The fraction of elements in each column to be set to zero.
- **`std`** (float, optional): The standard deviation of the normal distribution used to generate the non-zero values. Default is `0.01`.
- **`generator`** (optional): A generator for random number generation. Default is `None`.

**Method**

- **`sparse_(tensor, sparsity, std=0.01, generator=None)`**: Fills the input `tensor` as a sparse matrix.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of sparse_
tensor = tf.Variable(tf.zeros([3, 5]))
nn.sparse_(tensor, sparsity=0.1)
print(tensor)
```

# Model

These functions extend the `Model` class, allowing you to manage namespaces for layers, control freezing and unfreezing of layers, and set training or evaluation modes. Below are the descriptions and usage of each function:

---

## 1. **`training(self, flag=False)`**
   - **Function**: Sets the entire model or individual layers to training or evaluation mode.
   - **Parameters**:
     - `flag` (`bool`, optional): 
       - `False` (default): Sets the model to evaluation mode.
       - `True`: Sets the model to training mode.
   - **Effect**: Updates the `train_flag` attribute of all layers in `self.layer_list`. If a layer does not have a `train_flag` attribute, it uses the `training` attribute instead. 

   **Example**:
   ```python
   model.training(flag=True)
   ```
   **Result**: Sets all layers in the model to training mode by adjusting either `train_flag` or `training` attributes.

---

## 2. **`namespace(name=None)`**
   - **Function**: Assigns a namespace to layers in the model for tracking layers and parameters.
   - **Parameters**: 
     - `name` (`str`, optional): The name for the namespace of the model. If `None` is passed, no name is assigned to the model.
   - **Effect**: This function adds the layer name to `Model.name_list_`.

   **Example**:
   ```python
   model.namespace('block1')
   ```
   **Result**: The namespace for the model is set to `block1`.

---

## 3. **`freeze(self, name=None)`**
   - **Function**: Freezes the parameters of the model or a specific namespace, making them untrainable during training.
   - **Parameters**:
     - `name` (`str`, optional): Specifies the namespace to freeze. If `name` is `None`, it freezes the parameters in all namespaces.
   - **Effect**: This function iterates through all parameters in `self.layer_param` and sets them to be untrainable (`_trainable=False`).

   **Example**:
   ```python
   model.freeze('block1')
   ```
   **Result**: Freezes all layer parameters in the `block1` namespace, preventing them from being updated during training.

---

## 4. **`unfreeze(self, name=None)`**
   - **Function**: Unfreezes the parameters of the model or a specific namespace, making them trainable again.
   - **Parameters**:
     - `name` (`str`, optional): Specifies the namespace to unfreeze. If `name` is `None`, it unfreezes the parameters in all namespaces.
   - **Effect**: Iterates through all parameters in `self.layer_param` and sets them to be trainable (`_trainable=True`).

   **Example**:
   ```python
   model.unfreeze('block1')
   ```
   **Result**: Unfreezes all layer parameters in the `block1` namespace, allowing them to be updated during training.

---

## 5. **`eval(self, name=None, flag=True)`**
   - **Function**: Sets the model or specific namespaces to training or evaluation mode.
   - **Parameters**:
     - `name` (`str`, optional): Specifies the namespace to configure. If `name` is `None`, it iterates through all namespaces.
     - `flag` (`bool`, optional): 
       - `True`: Sets to evaluation mode (freezes layers).
       - `False`: Sets to training mode.
   - **Effect**: Controls the training state of each layer. When `flag=True`, the model is set to evaluation mode, and `train_flag=False`.

   **Example**:
   ```python
   model.eval('block1', flag=True)
   ```
   **Result**: Sets all layers in `block1` to evaluation mode (`train_flag=False`).

---

## Typical Use Cases:

- **Global training or evaluation mode**:
  - Use `training()` to set the entire model to training or evaluation mode. This is useful for switching between modes before starting the training or inference processes.
- **Naming layers in the model**: 
  - When you want to control different blocks independently, use `namespace()` to assign a unique name to different layers or modules.
- **Freezing or unfreezing layers**:
  - Use `freeze()` and `unfreeze()` to control which layers participate in gradient updates during training. For example, when fine-tuning a model, you may only want to unfreeze the top layers.
- **Setting training or evaluation modes**:
  - `eval()` allows you to easily switch between training and evaluation modes. During training, you may need to freeze certain layers or switch behaviors in some layers (like Batch Normalization, which behaves differently during training and inference).

These methods provide flexibility in managing complex models, particularly when freezing parameters and adjusting training strategies.
