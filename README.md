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

# constant_

The `constant_` function initializes a tensor with a constant value.

**Parameters**

- **`tensor`**: A `tf.Variable` to be filled with the constant value.
- **`val`**: The constant value to assign to all elements of the `tensor`. The value will be cast to the data type of the `tensor`.

**Method**

- **`constant_(tensor, val)`**: Fills the input `tensor` with the constant value `val`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example usage of constant_
tensor = tf.Variable(tf.zeros([3, 5]))
nn.constant_(tensor, val=3.14)
print(tensor)
```

# solve_triangular

The `solve_triangular` function solves a triangular system of linear equations using TensorFlow operations. It can handle both left- and right-sided systems and optionally assumes that the triangular matrix has unit diagonal elements.

**Parameters**

- **A**: A 2-dimensional `tf.Tensor` representing the triangular matrix.  
- **B**: A `tf.Tensor` representing the right-hand side matrix or vector.  
- **upper** (bool): Indicates whether the matrix `A` is upper triangular. If `True`, `A` is considered upper triangular; otherwise, it is considered lower triangular.  
- **left** (bool, optional): Determines the side of the equation to solve. If `True` (default), the function solves `A * X = B`; if `False`, it solves `X * A = B`.  
- **unitriangular** (bool, optional): If set to `True`, the function assumes `A` is unit triangular, meaning its diagonal elements are all ones. In this case, the diagonal of `A` is replaced with ones before solving. Default is `False`.

**Method**

- **solve_triangular(A, B, *, upper, left=True, unitriangular=False)**:  
  1. If `unitriangular` is `True`, replaces the diagonal of `A` with ones.  
  2. If `left` is `True`, it solves the system `A * X = B` using TensorFlow's `tf.linalg.triangular_solve`, with the `lower` parameter set based on the value of `upper`.  
  3. If `left` is `False`, it solves the system `X * A = B` by transposing `A` and `B`, solving the transposed system, and then transposing the result back.  
  4. Returns the solution tensor `X`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Example: Solving A * X = B where A is lower triangular.
A = tf.constant([[2.0, 0.0],
                 [3.0, 1.0]])
B = tf.constant([4.0, 7.0])

# Solve the system assuming A is lower triangular (upper=False)
X = nn.solve_triangular(A, B, upper=False)
print(X)

# Example: Solving X * A = B for an upper triangular A.
A = tf.constant([[2.0, 3.0],
                 [0.0, 1.0]])
B = tf.constant([[4.0, 5.0],
                 [6.0, 7.0]])

# Solve the system from the right side (left=False)
X = nn.solve_triangular(A, B, upper=True, left=False)
print(X)
```

# sparse_mask

The `sparse_mask` function constructs a new sparse tensor by using the indices and dense shape from a provided sparse tensor (`mask_sparse`) while extracting the corresponding values from a dense tensor (`dense_tensor`).

**Parameters**

- **dense_tensor**: A `tf.Tensor` containing the source values.
- **mask_sparse**: A `tf.sparse.SparseTensor` whose `indices` and `dense_shape` determine the positions in `dense_tensor` from which to gather values.

**Method**

- **sparse_mask(dense_tensor, mask_sparse)**:  
  1. Retrieves the indices from `mask_sparse`.
  2. Uses `tf.gather_nd` to extract the corresponding values from `dense_tensor`.
  3. Returns a new `tf.sparse.SparseTensor` with the gathered values and the same indices and dense shape as `mask_sparse`.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create a dense tensor
dense = tf.constant([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Create a sparse mask with specific indices
indices = tf.constant([[0, 1], [2, 2]])
mask_sparse = tf.sparse.SparseTensor(indices=indices, values=[0, 0], dense_shape=[3, 3])

# Construct a new sparse tensor using the mask from the dense tensor
sparse_result = nn.sparse_mask(dense, mask_sparse)
print("Sparse result:")
print(tf.sparse.to_dense(sparse_result))
```

# nan_to_num

The `nan_to_num` function replaces all `NaN` values in a tensor with a specified numeric value. It optionally allows writing the result to an output tensor.

**Parameters**

- **tensor**: A `tf.Tensor` containing the values to be processed.
- **nan** (float, optional): The value to replace any `NaN` values in the `tensor`. Default is `0.0`.
- **out** (optional): A `tf.Variable` to which the resulting tensor will be assigned. If provided, the function assigns the result to `out` and returns it; otherwise, it returns a new tensor with the replacements.

**Method**

- **`nan_to_num(tensor, nan=0.0, out=None)`**:  
  1. Uses `tf.math.is_nan` to identify `NaN` values within the `tensor`.
  2. Applies `tf.where` to substitute these `NaN` values with the specified `nan` parameter.
  3. If the `out` parameter is provided, assigns the resulting tensor to `out` and returns it; otherwise, returns the new tensor directly.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create a tensor with some NaN values
tensor = tf.constant([1.0, float('nan'), 3.0, float('nan')])

# Replace NaNs with 0.0 (default behavior)
result = nn.nan_to_num(tensor)
print("Result with default replacement (0.0):", result)

# Replace NaNs with a different value and assign to an existing variable
output_tensor = tf.Variable(tf.zeros_like(tensor))
nn.nan_to_num(tensor, nan=-1.0, out=output_tensor)
print("Result with replacement (-1.0) in output tensor:", output_tensor)
```

# coalesce_sparse

The `coalesce_sparse` function merges duplicate entries in a sparse tensor by summing their values, resulting in a properly “coalesced” `tf.SparseTensor`.

**Parameters**

- **`sp`** (`tf.SparseTensor`):  
  A sparse tensor potentially containing duplicate indices.

**Returns**

- **`tf.SparseTensor`**:  
  A new sparse tensor with:
  - **`indices`**: Unique coordinates from `sp.indices`.
  - **`values`**: Summed values for each unique coordinate.
  - **`dense_shape`**: Same as `sp.dense_shape`.

**Method**

- **`coalesce_sparse(sp: tf.SparseTensor) -> tf.SparseTensor`**  
  1. Casts `sp.dense_shape` to `int64`.  
  2. Constructs **multipliers** for row-major linear indexing via the cumulative product of `dense_shape[1:]` and a trailing 1.  
  3. Converts N‑D indices to 1‑D **linear indices** by dotting with `multipliers`.  
  4. Uses `tf.unique` to extract **unique linear indices** and **segment IDs** mapping each original index to its unique group citeturn1search0.  
  5. Applies `tf.math.unsorted_segment_sum` to **sum** `sp.values` across each segment ID citeturn2search0.  
  6. Converts unique linear indices back to N‑D indices with `tf.unravel_index` and stacks them.  
  7. Returns a new `tf.SparseTensor` constructed from these coalesced indices and summed values, preserving the original shape.

> **Merging duplicates** in sparse tensors ensures correct aggregation of values when the same coordinate appears multiple times citeturn0search0.

**Example Usage**

```python
import tensorflow as tf
from Note import nn

# Create a SparseTensor with duplicate indices
indices = tf.constant([[0, 1], [0, 1], [1, 2]])
values = tf.constant([3.0, 4.0, 5.0])
dense_shape = [3, 4]
sp = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

# Coalesce duplicates
coalesced_sp = nn.coalesce_sparse(sp)
print(tf.sparse.to_dense(coalesced_sp))
# Expected dense output:
# [[0, 7, 0, 0],
#  [0, 0, 5, 0],
#  [0, 0, 0, 0]]
```
