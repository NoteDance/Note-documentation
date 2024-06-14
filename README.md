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
