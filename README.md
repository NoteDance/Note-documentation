# conv2d
This module implements a 2D convolutional layer, which can apply a set of filters to an input tensor and produce a feature map. The usage of this module is as follows:

- First, create an instance of the conv2d class, and specify the input size, the number of output filters, the kernel size, and other optional parameters such as activation function, weight initializer, bias initializer, use bias, strides, padding mode, and data format.
- Second, call the output method of the instance, and pass the input tensor as the data argument. You can also specify the dilation rate of the convolution operation.
- The output method will return a tensor of shape [batch_size, height, width, filters], which is the 2D convolution output.

For example:

```python
# Create a 2D convolution layer with 32 output filters, 3x3 kernel size, ReLU activation
conv2d = conv2d(input_size=28, filters=32, kernel_size=[3, 3], activation='relu')
# Apply the 2D convolution layer to a batch of input images of shape [64, 28, 28]
input_images = tf.random.normal([64, 28, 28])
output_images = conv2d.output(input_images)
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

- First, create an instance of the dense class, and specify the output size, the input size, the weight initializer, the bias initializer, the activation function, the data type, and the use bias flag. The weight shape should be a list of two integers: [input_size, output_size]. The activation function can be any callable object that takes a tensor as input and returns a tensor as output. The use bias flag indicates whether to add a bias term to the linear transformation or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, input_size], where batch_size is the number of samples in a batch and input_size is the dimension of the input features.
- The output method will return a tensor of shape [batch_size,
  output_size], which is the dense output after applying the linear transformation and the activation function.

For example:

```python
# Create a dense layer with 32 output units and sigmoid activation
dense_layer = dense(32, 16, activation='sigmoid')
# Apply the dense layer to a batch of input data of shape [64, 16]
input_data = tf.random.normal([64, 16])
output_data = dense_layer.output(data=input_data)
# The output_data will have a shape of [64, 32]
```

# GRU
This module implements a gated recurrent unit (GRU) layer, which can process the input data in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the GRU class, and specify the output size, the input size, the weight initializer, the bias initializer, the data type, the return sequence flag, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformations or not. The activation functions should be callable objects that take a tensor as input and return a tensor as output.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the GRU output after applying the update gate and reset gate to the input data and the previous hidden state.

For example:

```python
# Create a GRU layer with 32 hidden units and tanh activation
gru_layer = GRU(32, 16)
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

# LSTM
This module implements a long short-term memory (LSTM) layer, which can process the input data in a sequential manner and learn long-term dependencies. The usage of this module is as follows:

- First, create an instance of the LSTM class, and specify the output size, the input size, the weight initializer, the bias initializer, the data type, the return sequence flag, the use bias flag, and the activation functions. The weight shape should be a list of two integers: [input_size, hidden_size]. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformations or not. The activation functions should be callable objects that take a tensor as input and return a tensor as output.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the LSTM output after applying the input gate, forget gate, output gate and cell state update to the input data and the previous hidden state and cell state.

For example:

```python
# Create an LSTM layer with 32 hidden units
lstm_layer = LSTM(32, 16)
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

- First, create an instance of the multihead_attention class, and specify the output size, the input size, the number of heads, and other optional parameters such as weight initializer, bias initializer, data type, and use bias flag. The weight shape should be a list of two integers: [input_size, hidden_size]. The number of heads should be a positive integer that can divide the hidden size. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, call the output method of the instance, and pass the input data as the data1 argument. Optionally, you can also pass another input data as the data2 argument, which will be used as the key and value for the attention computation. If data2 is not provided, data1 will be used as the query, key, and value. You can also pass a mask argument to mask out some tokens from the attention computation. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step. The mask should be a tensor of shape [batch_size, seq_length_q, seq_length_k], where seq_length_q is the number of time steps in data1 and seq_length_k is the number of time steps in data2 (or data1 if data2 is not provided).
- The output method will return a tuple of two tensors: output_data and attention_weights. The output_data is a tensor of shape [batch_size, seq_length_q,
  hidden_size], which is the multi-head attention output. The attention_weights is a tensor of shape [batch_size, num_heads, seq_length_q,
  seq_length_k], which is the scaled dot product attention weights for each head. The output_data is computed by applying query, key, and value projections to the input data, followed by scaled dot product attention with optional masking, concatenation of the attention outputs from each head, and output projection. The attention_weights is computed by applying query and key projections to the input data, followed by scaled dot product attention with optional masking.

For example:

```python
# Create a multi-head attention layer with 4 heads and 64 hidden size
multihead_attention_layer = multihead_attention(64, 4, 16)
# Apply the multi-head attention layer to a batch of input data of shape [32, 20, 16]
input_data = tf.random.normal([32, 20, 16])
output_data, attention_weights = multihead_attention_layer.output(data1=input_data)
# The output_data will have a shape of [32, 20, 64]
# The attention_weights will have a shape of [32, 4, 20, 20]
```

# RNN
This module implements a simple recurrent neural network (RNN) layer, which can process the input data in a sequential manner and learn short-term dependencies. The usage of this module is as follows:

- First, create an instance of the RNN class, and specify the output size, the input size, the weight initializer, the bias initializer, the activation function, the data type, the return sequence flag, and the use bias flag. The weight shape should be a list of two integers: [input_size, hidden_size]. The activation function should be a string that matches one of the keys in the activation_dict dictionary. The return sequence flag indicates whether to return the output at each time step or only at the last time step. The use bias flag indicates whether to add a bias term to the linear transformation or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size,
  seq_length,
  hidden_size] if the return sequence flag is True, or [batch_size,
  hidden_size] if the return sequence flag is False. This is the RNN output after applying the activation function to the linear combination of the input data and the previous hidden state.

For example:

```python
# Create an RNN layer with 32 hidden units and tanh activation
rnn_layer = RNN(32, 16, activation='tanh')
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

# Transformer
This module implements a Transformer layer, which can learn the self-attention and feed-forward features of the input data. The usage of this module is as follows:

- First, create an instance of the Transformer class, and specify the output size, the input size, the number of heads, and other optional parameters such as weight initializer, bias initializer, data type, and use bias flag. The weight shape should be a list of two integers: [input_size, hidden_size]. The number of heads should be a positive integer that can divide the hidden size. The use bias flag indicates whether to add a bias term to the linear transformations or not.
- Second, call the output method of the instance, and pass the input data as the data argument. The input data should be a tensor of shape [batch_size, seq_length, input_size], where batch_size is the number of samples in a batch, seq_length is the number of time steps in a sequence, and input_size is the dimension of the input features at each time step.
- The output method will return a tensor of shape [batch_size, seq_length,
  hidden_size], which is the Transformer output. The output is computed by applying multi-head self-attention and feed-forward network to the input data, followed by layer normalization and residual connection.

For example:

```python
# Create a Transformer layer with 8 heads and 128 hidden size
transformer_layer = Transformer(128, 8, 16)
# Apply the Transformer layer to a batch of input data of shape [64, 10, 16]
input_data = tf.random.normal([64, 10, 16])
output_data = transformer_layer.output(data=input_data)
# The output_data will have a shape of [64, 10, 128]
```
