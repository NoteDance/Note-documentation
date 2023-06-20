# AdaptiveConv2D
This module implements an adaptive convolution layer, which can adjust the output feature maps according to the input data. The usage of this module is as follows:

- First, create an instance of the AdaptiveConv2D class, and specify the input shape, the number of output filters, the kernel size, and other optional parameters such as activation function, kernel initializer, bias initializer, and use bias.
- Second, call the output method of the instance, and pass the input tensor as the data argument. You can also specify the strides and padding mode of the convolution operation.
- The output method will return a tensor of shape [batch_size, height, width, filters], which is the adaptive convolution output.

For example:

```python
# Create an adaptive convolution layer with 32 output filters, 3x3 kernel size, and ReLU activation
adaptive_conv = AdaptiveConv2D(input_shape=[28, 28, 1], filters=32, kernel_size=[3, 3], activation='ReLU')
# Apply the adaptive convolution layer to a batch of input images of shape [64, 28, 28, 1]
input_images = tf.random.normal([64, 28, 28, 1])
output_images = adaptive_conv.output(data=input_images)
# The output_images will have a shape of [64, 26, 26, 32]
```

# attention
This module implements an attention layer, which can compute the context vector and the attention weights based on the encoder hidden states and the decoder hidden state. The usage of this module is as follows:

- First, create an instance of the attention class, and specify the weight shape, the weight initializer, and the data type.
- Second, call the output method of the instance, and pass the encoder hidden states and the decoder hidden state as the en_h and de_h arguments. You can also pass the score_en_h argument, which is the projection of the encoder hidden states by a query weight matrix. If you don't pass this argument, it will be computed internally.
- The output method will return a tuple of three tensors: context_vector, score_en_h, and attention_weights. The context_vector is a tensor of shape [batch_size, hidden_size], which is the weighted sum of the encoder hidden states. The score_en_h is a tensor of shape [batch_size, seq_len, hidden_size], which is the projection of the encoder hidden states by a query weight matrix. The attention_weights is a tensor of shape [batch_size, seq_len], which is the softmax normalized score for each encoder hidden state.

For example:

```python
# Create an attention layer with weight shape [64, 128]
attention_layer = attention(weight_shape=[64, 128])
# Apply the attention layer to a batch of encoder hidden states of shape [32, 10, 64] and a decoder hidden state of shape [32, 64]
encoder_hidden_states = tf.random.normal([32, 10, 64])
decoder_hidden_state = tf.random.normal([32, 64])
context_vector, score_en_h, attention_weights = attention_layer.output(en_h=encoder_hidden_states, de_h=decoder_hidden_state)
# The context_vector will have a shape of [32, 128]
# The score_en_h will have a shape of [32, 10, 128]
# The attention_weights will have a shape of [32, 10]
```

# conv2d
This module implements a convolutional layer, which can apply a set of filters to the input data and produce a set of feature maps. The usage of this module is as follows:

- First, create an instance of the conv2d class, and specify the weight shape, the weight initializer, the bias initializer, the activation function, the data type, and the use bias flag. The weight shape should be a list of four integers: [filter_height, filter_width, in_channels, out_channels]. The activation function can be any callable object that takes a tensor as input and returns a tensor as output. The use bias flag indicates whether to add a bias term to the convolution output or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, height, width, channels] if the data format is 'NHWC', or [batch_size, channels, height, width] if the data format is 'NCHW'. You can also specify other arguments such as strides, padding, data format, and dilations. The strides argument should be a list of four integers: [batch_stride, height_stride, width_stride, channel_stride]. The padding argument should be either 'VALID' or 'SAME'. The data format argument should be either 'NHWC' or 'NCHW'. The dilations argument should be a list of four integers: [batch_dilation, height_dilation, width_dilation,
  channel_dilation].
- The output method will return a tensor of shape [batch_size, out_height, out_width,
  out_channels] if the data format is 'NHWC', or [batch_size,
  out_channels, out_height,
  out_width] if the data format is 'NCHW'. This is the convolution output after applying the activation function and adding the bias term (if use bias is True).

For example:

```python
# Create a convolutional layer with 32 filters of size 3x3 and ReLU activation
conv_layer = conv2d(weight_shape=[3, 3, 1, 32], activation=tf.nn.relu)
# Apply the convolutional layer to a batch of input images of shape [64, 28, 28, 1]
input_images = tf.random.normal([64, 28, 28, 1])
output_images = conv_layer.output(data=input_images,strides=[1, 1, 1, 1],padding='SAME')
# The output_images will have a shape of [64, 28, 28, 32]
```

# capsule
This module implements a capsule layer, which can learn to encode the part-whole relationships and pose information of the input data into a set of output vectors. The usage of this module is as follows:

- First, create an instance of the capsule class, and specify the input shape, the number of output capsules, the dimension of output capsules, and other optional parameters such as routings, weight initializer, and data type.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data can be either two-dimensional (batch_size, input_dim_capsules), three-dimensional (batch_size, input_num_capsules, input_dim_capsules), or four-dimensional (batch_size, height, width, channels). The output method will automatically reshape the input data to match the expected shape.
- The output method will return a tensor of shape (batch_size, num_capsules, dim_capsules), which is the output capsules. Each output capsule is a vector with length between 0 and 1, which represents the probability and pose of a part or an entity.

For example:

```python
# Create a capsule layer with 10 output capsules and 16 dimensions for each output capsule
capsule_layer = capsule(input_shape=[64, 1152, 8], num_capsules=10, dim_capsules=16)
# Apply the capsule layer to a batch of input data of shape [64, 1152, 8]
input_data = tf.random.normal([64, 1152, 8])
output_data = capsule_layer.output(data=input_data)
# The output_data will have a shape of [64, 10, 16]
```

# dense
This module implements a dense layer, which can apply a linear transformation and an optional activation function to the input data. The usage of this module is as follows:

- First, create an instance of the dense class, and specify the weight shape, the weight initializer, the bias initializer, the activation function, the data type, and the use bias flag. The weight shape should be a list of two integers: [input_size, output_size]. The activation function can be any callable object that takes a tensor as input and returns a tensor as output. The use bias flag indicates whether to add a bias term to the linear transformation or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, input_size], where batch_size is the number of samples in a batch and input_size is the dimension of the input features.
- The output method will return a tensor of shape [batch_size,
  output_size], which is the dense output after applying the linear transformation and the activation function.

For example:

```python
# Create a dense layer with 32 output units and sigmoid activation
dense_layer = dense(weight_shape=[16, 32], activation=tf.nn.sigmoid)
# Apply the dense layer to a batch of input data of shape [64, 16]
input_data = tf.random.normal([64, 16])
output_data = dense_layer.output(data=input_data)
# The output_data will have a shape of [64, 32]
```

# FAVOR_attention
This module implements a custom layer that performs FAVOR+ attention, which is a fast and memory-efficient approximation of softmax attention. The usage of this module is as follows:

- First, create an instance of the FAVOR_attention class, and specify the dimension of the input and output vectors, the number of attention heads, the number of random features, and other optional parameters such as weight initializer and data type. The number of random features should be divisible by the number of attention heads. The dimension should be divisible by the number of attention heads as well.
- Second, call the output method of the instance, and pass the input data as the data1 argument. The input data should be a tensor of shape [batch_size, seq_length, dim], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and dim is the dimension of the input and output vectors. You can also pass another input data as the data2 argument, which will be used to compute the key and value vectors instead of data1.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  dim * nb_heads], which is the FAVOR+ attention output. The output is computed by applying multi-head attention with random feature matrices to the input data.

For example:

```python
# Create a FAVOR+ attention layer with 8 heads and 64 random features
favor_attention_layer = FAVOR_attention(dim=16, nb_heads=8, nb_random_features=64)
# Apply the FAVOR+ attention layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = favor_attention_layer.output(data1=input_data)
# The output_data will have a shape of [64, 10, 128]
```

# GRU
This module implements a gated recurrent unit (GRU) layer, which can process the input data in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the GRU class, and specify the weight shape, the weight initializer, the bias initializer, the data type, the return sequence flag, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformations or not. The activation functions should be callable objects that take a tensor as input and return a tensor as output.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the GRU output after applying the update gate and reset gate to the input data and the previous hidden state.

For example:

```python
# Create a GRU layer with 32 hidden units and tanh activation
gru_layer = GRU(weight_shape=[16, 32], activation2=tf.nn.tanh)
# Apply the GRU layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = gru_layer.output(data=input_data)
# The output_data will have a shape of [64, 32]
```

# GRUCell
This module implements a gated recurrent unit (GRU) cell, which can process the input data and the previous state in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the GRUCell class, and specify the weight shape, the weight initializer, the bias initializer, the data type, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The activation functions should be callable objects that take a tensor as input and return a tensor as output. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, call the output method of the instance, and pass the input data and the previous state as the data and state arguments. The input data should be a tensor of shape [batch_size, input_size], where batch_size is the number of samples in a batch and input_size is the dimension of the input features. The previous state should be a tensor of shape [batch_size,
  hidden_size], where hidden_size is the dimension of the hidden state.
- The output method will return a tuple of two tensors: output and new_state. The output is a tensor of shape [batch_size,
  hidden_size], which is the output of the cell at the current time step. The new_state is a tensor of shape [batch_size,
  hidden_size], which is the updated hidden state for the next time step.

For example:

```python
# Create a GRU cell with 32 hidden units
gru_cell = GRUCell(weight_shape=[16, 32])
# Apply the GRU cell to a batch of input data of shape [64, 16] and a previous state of shape [64, 32]
input_data = tf.random.normal([64, 16])
prev_state = tf.random.normal([64, 32])
output_data, new_state = gru_cell.output(data=input_data,state=prev_state)
# The output_data will have a shape of [64, 32]
# The new_state will have a shape of [64, 32]
```

# Linear_attention
This module implements a linear attention layer, which can approximate the softmax attention mechanism with linear complexity using kernel methods. The usage of this module is as follows:

- First, create an instance of the Linear_attention class, and specify the dimension of the input and output vectors, the number of attention heads, the kernel function, and the kernel approximation method. The dimension should be divisible by the number of attention heads. The kernel function should be a callable object that takes two tensors as input and returns a tensor as output. The kernel approximation method should be either 'low_rank' or 'exact'.
- Second, call the output method of the instance, and pass the input data as the data1 argument. The input data should be a tensor of shape [batch_size, seq_length, dim], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and dim is the dimension of the input and output vectors. You can also pass another input data as the data2 argument, which will be used to compute the key and value vectors instead of data1. The data2 argument should have the same shape as data1.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  dim], which is the linear attention output. The output is computed by applying multi-head attention with kernel functions and kernel approximations to the input data.

For example:

```python
# Create a linear attention layer with 8 heads and Gaussian kernel
linear_attention_layer = Linear_attention(dim=16, num_heads=8, kernel_function='gaussian', kernel_approximation='low_rank')
# Apply the linear attention layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = linear_attention_layer.output(data1=input_data)
# The output_data will have a shape of [64, 10, 16]
```

# Linformer
This module defines a Linformer class that implements a linear transformer layer. A linear transformer layer is a variant of the standard transformer layer that uses linear attention instead of softmax attention. Linear attention is more efficient and scalable than softmax attention, especially for long sequences. The usage of this module is as follows:

- First, create an instance of the Linformer class, and specify the dimension, the number of heads, and other optional parameters such as kernel function and kernel approximation. The dimension should be a positive integer that represents the input and output size. The number of heads should be a positive integer that can divide the dimension. The kernel function should be either 'gaussian' or a function, and the kernel approximation should be either 'low_rank' or 'random_features'.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, dimension], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and dimension is the size of the input features at each time step.
- The output method will return a tensor of shape [batch_size, seq_length,
  dimension], which is the Linformer output. The output is computed by applying multi-head linear attention and feed-forward network to the input data, followed by layer normalization and residual connection.

For example:

```python
# Create a Linformer layer with 4 heads and 64 dimension
linformer_layer = Linformer(dim=64, num_heads=4)
# Apply the Linformer layer to a batch of input data of shape [32, 20, 64]
input_data = tf.random.normal([32, 20, 64])
output_data = linformer_layer.output(data=input_data)
# The output_data will have a shape of [32, 20, 64]
```

# Longformer
This module defines a Longformer class that implements a custom layer that uses local and global attention. Local and global attention are variants of the standard self-attention that can handle long sequences more efficiently and selectively. The usage of this module is as follows:

- First, create an instance of the Longformer class, and specify the dimension, the number of heads, the window size, and the global tokens. The dimension should be a positive integer that represents the input and output size. The number of heads should be a positive integer that can divide the dimension. The window size should be a positive integer that determines how many tokens on each side of a token can attend to it. The global tokens should be a list of integers that indicate the indices of the tokens that use global attention.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, dimension], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and dimension is the size of the input features at each time step.
- The output method will return a tensor of shape [batch_size, seq_length,
  dimension], which is the Longformer output. The output is computed by applying multi-head local and global attention and feed-forward network to the input data, followed by layer normalization and residual connection.

For example:

```python
# Create a Longformer layer with 4 heads, 64 dimension, 8 window size, and [0, 9] global tokens
longformer_layer = Longformer(dim=64, num_heads=4, window_size=8, global_tokens=[0, 9])
# Apply the Longformer layer to a batch of input data of shape [32, 20, 64]
input_data = tf.random.normal([32, 20, 64])
output_data = longformer_layer.output(data=input_data)
# The output_data will have a shape of [32, 20, 64]
```

# LSTM
This module implements a long short-term memory (LSTM) layer, which can process the input data in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the LSTM class, and specify the weight shape, the weight initializer, the bias initializer, the data type, the return sequence flag, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformations or not. The activation functions should be callable objects that take a tensor as input and return a tensor as output.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the LSTM output after applying the input gate, forget gate, output gate and cell state update to the input data and the previous hidden state and cell state.

For example:

```python
# Create an LSTM layer with 32 hidden units
lstm_layer = LSTM(weight_shape=[16, 32])
# Apply the LSTM layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = lstm_layer.output(data=input_data)
# The output_data will have a shape of [64, 32]
```

# LSTMCell
This module implements a long short-term memory (LSTM) cell, which can process the input data and the previous state in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the LSTMCell class, and specify the weight shape, the weight initializer, the bias initializer, the data type, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The activation functions should be callable objects that take a tensor as input and return a tensor as output. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, call the output method of the instance, and pass the input data and the previous state as the data and state arguments. The input data should be a tensor of shape [batch_size, input_size], where batch_size is the number of samples in a batch and input_size is the dimension of the input features. The previous state should be a tensor of shape [batch_size,
  hidden_size], where hidden_size is the dimension of the hidden state.
- The output method will return a tuple of two tensors: output and new_state. The output is a tensor of shape [batch_size,
  hidden_size], which is the output of the cell at the current time step. The new_state is a tensor of shape [batch_size,
  hidden_size], which is the updated hidden state for the next time step.

For example:

```python
# Create an LSTM cell with 32 hidden units
lstm_cell = LSTMCell(weight_shape=[16, 32])
# Apply the LSTM cell to a batch of input data of shape [64, 16] and a previous state of shape [64, 32]
input_data = tf.random.normal([64, 16])
prev_state = tf.random.normal([64, 32])
output_data, new_state = lstm_cell.output(data=input_data, state=prev_state)
# The output_data will have a shape of [64, 32]
# The new_state will have a shape of [64, 32]
```

# multihead_attention
This module defines a multihead_attention class that implements a multi-head self-attention layer. A multi-head self-attention layer is a sublayer of the standard transformer layer that can learn the relevance and dependency of different tokens in a sequence. The usage of this module is as follows:

- First, create an instance of the multihead_attention class, and specify the weight shape, the number of heads, and other optional parameters such as weight initializer, bias initializer, data type, and use bias flag. The weight shape should be a list of two integers: [input_size, hidden_size]. The number of heads should be a positive integer that can divide the hidden size. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, call the output method of the instance, and pass the input data as the data1 argument. Optionally, you can also pass another input data as the data2 argument, which will be used as the key and value for the attention computation. If data2 is not provided, data1 will be used as the query, key, and value. You can also pass a mask argument to mask out some tokens from the attention computation. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step. The mask should be a tensor of shape [batch_size, seq_length_q, seq_length_k], where seq_length_q is the number of time steps in data1 and seq_length_k is the number of time steps in data2 (or data1 if data2 is not provided).
- The output method will return a tuple of two tensors: output_data and attention_weights. The output_data is a tensor of shape [batch_size, seq_length_q,
  hidden_size], which is the multi-head attention output. The attention_weights is a tensor of shape [batch_size, num_heads, seq_length_q,
  seq_length_k], which is the scaled dot product attention weights for each head. The output_data is computed by applying query, key, and value projections to the input data, followed by scaled dot product attention with optional masking, concatenation of the attention outputs from each head, and output projection. The attention_weights is computed by applying query and key projections to the input data, followed by scaled dot product attention with optional masking.

For example:

```python
# Create a multi-head attention layer with 4 heads and 64 hidden size
multihead_attention_layer = multihead_attention(weight_shape=[16, 64], num_heads=4)
# Apply the multi-head attention layer to a batch of input data of shape [32, 20, 16]
input_data = tf.random.normal([32, 20, 16])
output_data, attention_weights = multihead_attention_layer.output(data1=input_data)
# The output_data will have a shape of [32, 20, 64]
# The attention_weights will have a shape of [32, 4, 20, 20]
```

# Performer
This module defines a Performer class that implements a custom layer that uses a performer block. A performer block is a variant of the standard transformer block that uses FAVOR+ attention instead of softmax attention. FAVOR+ attention is a fast and scalable approximation of softmax attention that uses random features. The usage of this module is as follows:

- First, create an instance of the Performer class, and specify the dimension, the number of heads, the number of random features, and other optional parameters such as weight initializer, bias initializer, activation function, data type, and use bias flag. The dimension should be a positive integer that represents the input and output size. The number of heads should be a positive integer that can divide the dimension. The number of random features should be a positive integer that determines the accuracy of the FAVOR+ attention approximation. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, dimension], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and dimension is the size of the input features at each time step.
- The output method will return a tensor of shape [batch_size, seq_length,
  dimension], which is the Performer output. The output is computed by applying multi-head FAVOR+ attention and feed-forward network to the input data, followed by layer normalization and residual connection.

For example:

```python
# Create a Performer layer with 4 heads, 64 dimension, 256 random features
performer_layer = Performer(dim=64, nb_heads=4, nb_random_features=256)
# Apply the Performer layer to a batch of input data of shape [32, 20, 64]
input_data = tf.random.normal([32, 20, 64])
output_data = performer_layer.output(data=input_data)
# The output_data will have a shape of [32, 20, 64]
```

# Ripple_attention
This module implements a ripple attention layer, which can compute the attention score using a ripple function that normalizes the scaled dot product by the cumulative sum. The usage of this module is as follows:

- First, create an instance of the Ripple_attention class, and specify the dimension of the input and output vectors, the number of attention heads, and other optional parameters such as weight initializer and data type. The dimension should be divisible by the number of attention heads.
- Second, call the output method of the instance, and pass the input data as the data1 argument. The input data should be a tensor of shape [batch_size, seq_length, dim], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and dim is the dimension of the input and output vectors. You can also pass another input data as the data2 argument, which will be used to compute the key and value vectors instead of data1.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  dim], which is the ripple attention output. The output is computed by applying multi-head attention with ripple function to the input data.

For example:

```python
# Create a ripple attention layer with 8 heads
ripple_attention_layer = Ripple_attention(dim=16, num_heads=8)
# Apply the ripple attention layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = ripple_attention_layer.output(data1=input_data)
# The output_data will have a shape of [64, 10, 16]
```

# RNN
This module implements a simple recurrent neural network (RNN) layer, which can process the input data in a sequential manner and learn short-term dependencies. The usage of this module is as follows:

- First, create an instance of the RNN class, and specify the weight shape, the weight initializer, the bias initializer, the activation function, the data type, the return sequence flag, and the use bias flag. The weight shape should be a list of two integers: [input_size, hidden_size]. The activation function should be a string that matches one of the keys in the activation_dict dictionary. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformation or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the RNN output after applying the activation function to the linear combination of the input data and the previous hidden state.

For example:

```python
# Create an RNN layer with 32 hidden units and tanh activation
rnn_layer = RNN(weight_shape=[16, 32], activation='tanh')
# Apply the RNN layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = rnn_layer.output(data=input_data)
# The output_data will have a shape of [64, 32]
```

# self_attention
This module implements a self-attention layer, which can learn the relevance and importance of the input data at different positions. The usage of this module is as follows:

- First, create an instance of the self_attention class, and specify the weight shape, the weight initializer, and the data type. The weight shape should be a list of two integers: [input_size, hidden_size]. The hidden size should be divisible by the number of attention heads.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step. You can also pass the number of attention heads as the a argument, and an optional mask tensor as the mask argument. The mask tensor should have a shape of [batch_size, num_heads, seq_length_q,
  seq_length_k], where seq_length_q and seq_length_k are the lengths of query and key sequences. The mask tensor can be used to prevent attention to some unwanted positions (such as padding tokens).
- The output method will return a tuple of two tensors: output and attention_weights. The output is a tensor of shape [batch_size,
  seq_length,
  hidden_size], which is the weighted sum of the value vectors for each position. The attention_weights is a tensor of shape [batch_size,
  num_heads,
  seq_length_q,
  seq_length_k], which is the normalized score for each pair of query and key positions.

For example:

```python
# Create a self-attention layer with 4 heads and 64 hidden size
self_attention_layer = self_attention(weight_shape=[16, 64])
# Apply the self-attention layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data, attention_weights = self_attention_layer.output(data=input_data,a=4)
# The output_data will have a shape of [64, 10, 64]
# The attention_weights will have a shape of [64, 4, 10, 10]
```

# sparse_attention
This module implements a sparse attention layer, which can compute the attention score using a scaled dot product and apply a sparsity mask to reduce the computation and memory cost. The usage of this module is as follows:

- First, create an instance of the sparse_attention class, and specify the weight shape, the weight initializer, the data type, the mask mode, and the mask parameters. The weight shape should be a list of two integers: [input_size, output_size]. The mask mode should be either None or one of the following options: 'local_window', 'block', or 'routing'. The mask parameters should be a dictionary that contains the necessary parameters for the chosen mask mode. For example, for 'local_window' mode, the mask parameters should have a key 'window_size' that specifies the size of the local window. For 'block' mode, the mask parameters should have a key 'block_size' that specifies the size of the block. For 'routing' mode, the mask parameters should have a key 'top_k' that specifies the number of top similarity scores to keep for each query.
- Second, call the output method of the instance, and pass the input data as the data1 argument. The input data should be a tensor of shape [batch_size, seq_length1, input_size], where batch_size is the number of samples in a batch, seq_length1 is the number of time steps in the first sequence, and input_size is the dimension of the input features. You can also pass another input data as the data2 argument, which will be used to compute the key and value vectors instead of data1. The data2 argument should be a tensor of shape [batch_size,
  seq_length2,
  input_size], where seq_length2 is the number of time steps in the second sequence. You also need to pass an integer as the a argument, which is the number of attention heads to use.
- The output method will return a tuple of two tensors: output and attention_weights. The output is a tensor of shape [batch_size,
  seq_length1,
  output_size], which is the sparse attention output. The attention_weights is a tensor of shape [batch_size,
  num_heads,
  seq_length1,
  seq_length2], which is the sparse attention weights after applying softmax and sparsity mask.

For example:

```python
# Create a sparse attention layer with local window mask
sparse_attention_layer = sparse_attention(weight_shape=[16, 32], mask_mode='local_window', mask_params={'window_size': 3})
# Apply the sparse attention layer to a batch of input data of shape [64, 10, 16] with 4 heads
input_data = tf.random.normal([64, 10, 16])
output_data, attention_weights = sparse_attention_layer.output(data1=input_data, a=4)
# The output_data will have a shape of [64, 10, 32]
# The attention_weights will have a shape of [64, 4, 10, 10]
```

# Transformer
This module implements a Transformer layer, which can learn the self-attention and feed-forward features of the input data. The usage of this module is as follows:

- First, create an instance of the Transformer class, and specify the weight shape, the number of heads, and other optional parameters such as weight initializer, bias initializer, data type, and use bias flag. The weight shape should be a list of two integers: [input_size, hidden_size]. The number of heads should be a positive integer that can divide the hidden size. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size, seq_length,
  hidden_size], which is the Transformer output. The output is computed by applying multi-head self-attention and feed-forward network to the input data, followed by layer normalization and residual connection.

For example:

```python
# Create a Transformer layer with 8 heads and 128 hidden size
transformer_layer = Transformer(weight_shape=[16, 128], num_heads=8)
# Apply the Transformer layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = transformer_layer.output(data=input_data)
# The output_data will have a shape of [64, 10, 128]
```
