# ConvNeXt:
The ConvNeXt class is a class written in Python that implements the ConvNeXt model, which is a convolutional neural network that uses grouped convolutions and layer scaling to improve performance and efficiency. The ConvNeXt class has the following attributes and methods:

```python
ConvNeXt(model_type='base',drop_path_rate=0.0,layer_scale_init_value=1e-6,classes=1000,include_top=True,pooling=None)
```

- **model_type:** A string that indicates the type of the model, which can be one of 'tiny', 'small', 'base', 'large' or 'xlarge'. Different model types have different depths and projection dimensions.
- **drop_path_rate:** A float that indicates the probability of stochastic depth, i.e., the probability of each convolutional block being dropped. Stochastic depth can improve the generalization and robustness of the model.
- **layer_scale_init_value:** A float that indicates the initial value of layer scaling, i.e., the coefficient that each convolutional block's output is multiplied by. Layer scaling can stabilize the training and convergence of the model.
- **classes:** An integer that indicates the number of classes for the classification task. If include_top is True, the last layer of the model is a fully connected layer that outputs neurons equal to the number of classes.
- **include_top:** A boolean that indicates whether to include the top layer. If True, the last layer of the model is a fully connected layer that outputs neurons equal to the number of classes. If False, the last layer of the model is a global average pooling layer or a global max pooling layer, depending on the pooling parameter.
- **pooling:** A string or None that indicates the pooling method. If include_top is False, the last layer of the model is a pooling layer, depending on the pooling parameter. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **loss_object:** A TensorFlow object that indicates the loss function. The default loss function is categorical cross-entropy.
- **optimizer:** A parallel optimizer for Note. The default optimizer is Adam.
- **param:** A list that stores all the parameters (weights and biases) of the model.
- **km:** An integer that indicates the kernel mode.
- **build(dtype='float32'):** A method that builds the structure of the model. It accepts one parameter dtype, which indicates the data type, defaulting to 'float32'. This method creates all the convolutional blocks, downsampling blocks, normalization layers, pooling layers and fully connected layers required by the model and stores them in their respective attributes.
- **fp(data, p):** A method that performs forward propagation. It accepts two parameters data and p, which indicate the input data and process number respectively. This method passes the input data through all the layers of the model in turn and returns the output data.
- **loss(output, labels, p):** A method that calculates the loss value. It accepts three parameters output, labels and p, which indicate the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between the output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p):** A method that performs optimization update. It accepts two parameters gradient and p, which indicate the gradient value and process number respectively. This method uses the optimizer to update all parameters of the model according to gradient value and returns updated parameters.

# DenseNet121:
The DenseNet121 class is a Python class that implements the DenseNet-121 model, which is a type of convolutional neural network that uses dense blocks and transition layers to improve the feature extraction and efficiency of convolutional neural networks. The DenseNet121 class has the following attributes and methods:

```python
DenseNet121(growth_rate=32, compression_factor=0.5, num_classes=1000, include_top=True, pooling=None, dtype='float32')
```

- **growth_rate**: An int, indicating the number of filters added by each dense layer. A larger growth rate increases the model size and complexity.
- **compression_factor**: a float, indicating the compression factor for the transition layers. A smaller compression factor reduces the number of filters and the model size.
- **num_classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs num_classes neurons.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs num_classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **dtype**: A string or TensorFlow dtype object, indicating the data type for computation. The default value is 'float32', which corresponds to 32-bit floating point numbers.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build()**: a method, used to build the model's structure. This method creates all the convolutional layers, dense layers, batch normalization layers, average pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# DenseNet169:
The DenseNet169 class is a Python class that implements the DenseNet-169 model, which is a type of convolutional neural network that uses dense blocks and transition layers to improve the feature extraction and efficiency of convolutional neural networks. The DenseNet169 class has the following attributes and methods:

```python
DenseNet169(growth_rate=32, compression_factor=0.5, num_classes=1000, include_top=True, pooling=None, dtype='float32')
```

- **growth_rate**: An int, indicating the number of filters added by each dense layer. A larger growth rate increases the model size and complexity.
- **compression_factor**: a float, indicating the compression factor for the transition layers. A smaller compression factor reduces the number of filters and the model size.
- **num_classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs num_classes neurons.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs num_classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **dtype**: A string or TensorFlow dtype object, indicating the data type for computation. The default value is 'float32', which corresponds to 32-bit floating point numbers.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build()**: a method, used to build the model's structure. This method creates all the convolutional layers, dense layers, batch normalization layers, average pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# DenseNet201:
The DenseNet201 class is a Python class that implements the DenseNet-201 model, which is a type of convolutional neural network that uses dense blocks and transition layers to improve the feature extraction and efficiency of convolutional neural networks. The DenseNet201 class has the following attributes and methods:

```python
DenseNet201(growth_rate=32, compression_factor=0.5, num_classes=1000, include_top=True, pooling=None, dtype='float32')
```

- **growth_rate**: An int, indicating the number of filters added by each dense layer. A larger growth rate increases the model size and complexity.
- **compression_factor**: a float, indicating the compression factor for the transition layers. A smaller compression factor reduces the number of filters and the model size.
- **num_classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs num_classes neurons.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs num_classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **dtype**: A string or TensorFlow dtype object, indicating the data type for computation. The default value is 'float32', which corresponds to 32-bit floating point numbers.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build()**: a method, used to build the model's structure. This method creates all the convolutional layers, dense layers, batch normalization layers, average pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# EfficientNet:
The EfficientNet class is a Python class that implements the EfficientNet model, which is a type of convolutional neural network that uses a compound scaling method to balance the network depth, width and resolution. It also uses inverted residual blocks with depthwise separable convolutions to reduce the computational cost and parameter count. The model has several variants, such as B0 to B7, that differ in their scaling coefficients and input sizes. The EfficientNet class has the following attributes and methods:

```python
EfficientNet(
    input_shape,
    model_name='B0',
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation="swish",
    blocks_args="default",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    device='GPU',
    dtype='float32'
)
```

- **input_shape**: A tuple of three integers, indicating the shape of the input data, excluding the batch dimension.
- **model_name**: A string, indicating the variant of the model. The default value is 'B0', which corresponds to the base model. Other possible values are 'B1' to 'B7', which correspond to different scaling coefficients and input sizes.
- **drop_connect_rate**: A float, indicating the dropout rate for the drop connect layer. The default value is 0.2, which means 20% of the connections are randomly dropped.
- **depth_divisor**: An integer, indicating the divisor for rounding the filters and repeats. The default value is 8, which means the filters and repeats are rounded to the nearest multiple of 8.
- **activation**: A string, indicating the activation function for the model. The default value is "swish", which corresponds to the swish activation function. Other possible values are "relu", "sigmoid", "tanh", etc.
- **blocks_args**: A list of dictionaries or "default", indicating the arguments for each block of the model. Each dictionary contains the keys "kernel_size", "repeats", "filters_in", "filters_out", "expand_ratio", "id_skip", "strides", "se_ratio" and "conv_type", which correspond to different parameters for each block. If blocks_args is "default", the default arguments from the original paper are used.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **weights**: A string or None, indicating the initial weights for the model. If weights is "imagenet", the model is initialized with pre-trained weights on ImageNet dataset. If weights is None, the model is initialized with random weights. If weights is a path to a file, the model is initialized with weights from that file.
- **input_tensor**: A TensorFlow tensor or None, indicating the input tensor for the model. If input_tensor is None, a new input tensor is created with shape (None,) + input_shape.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **classifier_activation**: A string or None, indicating the activation function for the classifier layer. If include_top is True, this argument specifies the activation function for the last layer. The default value is "softmax", which corresponds to the softmax activation function. Other possible values are "sigmoid", "tanh", etc.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **dtype**: A string or TensorFlow dtype object, indicating the data type for computation. The default value is 'float32', which corresponds to 32-bit floating point numbers.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build()**: A method, used to build the model's structure. It accepts no arguments. This method creates all the convolutional layers, depthwise separable convolutional layers, inverted residual blocks, batch normalization layers, activation layers, dropout layers, drop connect layers, global pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# EfficientNetV2:
The EfficientNetV2 class is a Python class that implements the EfficientNetV2 model, which is a type of convolutional neural network that uses a compound scaling method to balance the network depth, width and resolution. It also uses inverted residual blocks with depthwise separable convolutions to reduce the computational cost and parameter count. The EfficientNetV2 model introduces some new features, such as fused MBConv blocks, progressive learning of feature maps, and self-training with noisy student. The model has several variants, such as B0 to M2, that differ in their scaling coefficients and input sizes. The EfficientNetV2 class has the following attributes and methods:

```python
EfficientNetV2(
    input_shape,
    model_name="efficientnetv2-b0",
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    min_depth=8,
    bn_momentum=0.9,
    activation="swish",
    blocks_args="default",
    include_top=True,
    weights="imagenet",
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
    device='GPU',
    dtype='float32'
)
```

- **input_shape**: A tuple of three integers, indicating the shape of the input data, excluding the batch dimension.
- **model_name**: A string, indicating the variant of the model. The default value is 'B0', which corresponds to the base model. Other possible values are 'B1' to 'M2', which correspond to different scaling coefficients and input sizes.
- **dropout_rate**: A float, indicating the dropout rate for the dropout layer. The default value is 0.2, which means 20% of the neurons are randomly dropped.
- **drop_connect_rate**: A float, indicating the dropout rate for the drop connect layer. The default value is 0.2, which means 20% of the connections are randomly dropped.
- **depth_divisor**: An integer, indicating the divisor for rounding the filters and repeats. The default value is 8, which means the filters and repeats are rounded to the nearest multiple of 8.
- **min_depth**: An integer or None, indicating the minimum depth for rounding the filters. The default value is 8, which means the minimum depth is 8. If None, no minimum depth is applied.
- **bn_momentum**: A float, indicating the momentum for the batch normalization layer. The default value is 0.9, which means 90% of the previous moving average is retained.
- **activation**: A string, indicating the activation function for the model. The default value is "swish", which corresponds to the swish activation function. Other possible values are "relu", "sigmoid", "tanh", etc.
- **blocks_args**: A list of dictionaries or "default", indicating the arguments for each block of the model. Each dictionary contains the keys "kernel_size", "num_repeat", "input_filters", "output_filters", "expand_ratio", "id_skip", "strides", "se_ratio" and "conv_type", which correspond to different parameters for each block. If blocks_args is "default", the default arguments from the original paper are used.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **weights**: A string or None, indicating the initial weights for the model. If weights is "imagenet", the model is initialized with pre-trained weights on ImageNet dataset. If weights is None, the model is initialized with random weights. If weights is a path to a file, the model is initialized with weights from that file.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **classifier_activation**: A string or None, indicating the activation function for the classifier layer. If include_top is True, this argument specifies the activation function for the last layer. The default value is "softmax", which corresponds to the softmax activation function. Other possible values are "sigmoid", "tanh", etc.
- **include_preprocessing**: A bool or None, indicating whether to include preprocessing layers for rescaling and normalization. If True, the model's first layer is a rescaling layer that scales the input data to a certain range, and the second layer is a normalization layer that normalizes the input data to have zero mean and unit variance. If False, no preprocessing layers are added. If None, the default preprocessing layers are used according to the model variant.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **dtype**: A string or TensorFlow dtype object, indicating the data type for computation. The default value is 'float32', which corresponds to 32-bit floating point numbers.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build()**: A method, used to build the model's structure. It accepts no arguments. This method creates all the convolutional layers, depthwise separable convolutional layers, inverted residual blocks, batch normalization layers, activation layers, dropout layers, drop connect layers, global pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# GPT2:
The GPT2 class is a Python class that implements the GPT-2 model, which is a type of transformer-based language model that can generate coherent and diverse texts on various topics and tasks. The GPT-2 model has several variants, such as small, medium, large and XL, that differ in their number of parameters and layers. The GPT2 class has the following attributes and methods:

```python
GPT2(one_hot=True)
```

- **one_hot**: A bool, indicating whether to use one-hot encoding for the labels. If True, the labels are converted to one-hot vectors before computing the loss. If False, the labels are used as indices for the logits.
- **norm**: A norm object, indicating the layer normalization layer for the model output.
- **block**: A dictionary, storing the block objects for each layer of the model. Each block object contains the attention layer, the feed-forward layer and the residual connections for that layer.
- **opt**: A TensorFlow object, indicating the optimizer. The default optimizer is Adam optimizer.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **flag**: An integer that indicates whether the model has been built or not.
- **fp(X, past=None)**: A method, used to perform forward propagation. It accepts two arguments `X` and `past`, indicating the input data and previous hidden states respectively. This method passes the input data through all the layers of the model and returns a dictionary with keys 'present' and 'logits'. The 'present' value is a tensor that contains the current hidden states of the model, which can be used as past for the next iteration. The 'logits' value is a tensor that contains the output logits of the model for each token in the input data.
- **loss(output, labels)**: A method, used to calculate the loss value. It accepts two arguments `output` and `labels`, indicating the output data and true labels respectively. This method uses the categorical crossentropy loss function to calculate the difference between output logits and true labels and returns the loss value. If one_hot is True, this method converts the labels to one-hot vectors before computing the loss.

# GPT2_:
The GPT2_ class is a Python class that implements the GPT-2 model, which is a type of transformer-based language model that can generate coherent and diverse texts on various topics and tasks. The GPT-2 model has several variants, such as small, medium, large and XL, that differ in their number of parameters and layers. The GPT2_ class has the following attributes and methods:

```python
GPT2_(one_hot=True, device='GPU')
```

- **one_hot**: A bool, indicating whether to use one-hot encoding for the labels. If True, the labels are converted to one-hot vectors before computing the loss. If False, the labels are used as indices for the logits.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **norm**: A norm object, indicating the layer normalization layer for the model output.
- **block**: A dictionary, storing the block objects for each layer of the model. Each block object contains the attention layer, the feed-forward layer and the residual connections for that layer.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **flag**: An integer that indicates whether the model has been built or not.
- **build()**: A method, used to build the model's structure. It accepts no arguments. This method creates all the weights and biases for the model, and initializes them with random values or pre-trained values if available.
- **fp(X, p=None, past=None)**: A method, used to perform forward propagation. It accepts three arguments `X`, `p` and `past`, indicating the input data, process number and previous hidden states respectively. This method passes the input data through all the layers of the model and returns a dictionary with keys 'present' and 'logits'. The 'present' value is a tensor that contains the current hidden states of the model, which can be used as past for the next iteration. The 'logits' value is a tensor that contains the output logits of the model for each token in the input data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the categorical crossentropy loss function to calculate the difference between output logits and true labels and returns the loss value. If one_hot is True, this method converts the labels to one-hot vectors before computing the loss.
- **GradientTape(data, labels, p)**: A method, used to calculate the gradient value. It accepts three arguments `data`, `labels` and `p`, indicating the input data, true labels and process number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. It returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# MobileNet:
The MobileNet class is a Python class that implements the MobileNet model, which is a type of convolutional neural network that uses depthwise separable convolutions and ReLU6 activation to reduce the computational cost and model size. The MobileNet class has the following attributes and methods:

```python
MobileNet(alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, pooling=None, classes=1000)
```

- **alpha**: A float, indicating the width multiplier that controls the number of filters in each layer. A smaller alpha reduces the number of filters and the model size.
- **depth_multiplier**: A float, indicating the depth multiplier that controls the number of depthwise convolution output channels. A smaller depth_multiplier reduces the number of channels and the model size.
- **dropout**: A float, indicating the dropout rate that is applied to the last layer before the classification layer. Dropout can improve the model's generalization and robustness.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the convolution blocks, depthwise convolution blocks, normalization layers, pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# MobileNetV2:
The MobileNetV2 class is a Python class that implements the MobileNetV2 model, which is a type of convolutional neural network that uses inverted residual blocks and linear bottlenecks to improve performance and efficiency. The MobileNetV2 class has the following attributes and methods:

```python
MobileNetV2(alpha=1.0,classes=1000,include_top=True,pooling=None)
```

- **alpha**: A float, indicating the width multiplier that controls the number of filters in each layer. A smaller alpha reduces the number of filters and the model size.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the convolution blocks, depthwise convolution blocks, normalization layers, pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# ResNetRS:
The ResNetRS class is a Python class that implements the ResNet-RS model, which is a type of convolutional neural network that uses residual blocks and stochastic depth to achieve state-of-the-art performance on image classification tasks. The ResNet-RS class has the following attributes and methods:

```python
ResNetRS(
            bn_momentum=0.0,
            bn_epsilon=1e-5,
            activation: str = "relu",
            se_ratio=0.25,
            dropout_rate=0.25,
            drop_connect_rate=0.2,
            include_top=True,
            block_args: List[Dict[str, int]] = None,
            model_name="resnet-rs-50",
            pooling=None,
            classes=1000,
            include_preprocessing=True,
    )
```

- **bn_momentum**: A float, indicating the momentum for the batch normalization layers.
- **bn_epsilon**: A float, indicating the epsilon for the batch normalization layers.
- **activation**: A string, indicating the activation function for the convolutional layers. The default activation function is ReLU.
- **se_ratio**: A float, indicating the ratio for the Squeeze and Excitation blocks. The default ratio is 0.25.
- **dropout_rate**: A float, indicating the dropout rate for the last layer before the classification layer. Dropout can improve the model's generalization and robustness.
- **drop_connect_rate**: A float, indicating the initial rate for the stochastic depth. Stochastic depth can improve the model's generalization and robustness by randomly dropping out blocks during training.
- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **block_args**: A list of dictionaries, indicating the arguments for each block group. Each dictionary contains the input filters and the number of repeats for each block group. The default block arguments are based on the model depth.
- **model_name**: A string, indicating the name of the ResNet-RS variant. The name determines the model depth and block arguments.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **include_preprocessing**: a bool, indicating whether to include preprocessing for the input data. If True, the input data will be rescaled and normalized according to ImageNet statistics.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the stem blocks, block groups and head blocks that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# VGG16:
The VGG16 class is a Python class that implements the VGG-16 model, which is a type of convolutional neural network that uses 16 layers of convolution, pooling and fully connected layers to achieve high performance on image classification tasks. The VGG16 class has the following attributes and methods:

```python
VGG16(include_top=True,pooling=None,classes=1000)
```

- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the convolutional layers, max pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.

# VGG19:
The VGG19 class is a Python class that implements the VGG-19 model, which is a type of convolutional neural network that uses 19 layers of convolution, pooling and fully connected layers to achieve high performance on image classification tasks. The VGG19 class has the following attributes and methods:

```python
VGG19(include_top=True,pooling=None,classes=1000)
```

- **include_top**: A bool, indicating whether to include the top layer for classification. If True, the model's last layer is a fully connected layer that outputs classes neurons. If False, the model's last layer is a global average pooling layer or a global max pooling layer, depending on the pooling argument.
- **pooling**: A string or None, indicating the pooling method. If include_top is False, the model's last layer is a pooling layer, depending on the pooling argument. If pooling is 'avg', global average pooling is used; if pooling is 'max', global max pooling is used; if pooling is None, no pooling is used.
- **classes**: An int, indicating the number of classes for the classification task. If include_top is True, the model's last layer is a fully connected layer that outputs classes neurons.
- **device**: A string, indicating the device to use for computation. The default value is 'GPU', which means using GPU if available. Other value is 'CPU'.
- **loss_object**: A TensorFlow object, indicating the loss function. The default loss function is categorical crossentropy loss.
- **optimizer**: A parallel optimizer for Note. The default optimizer is Adam.
- **param**: A list, storing all the parameters (weights and biases) of the model.
- **km**: An integer that indicates the kernel mode.
- **build(dtype='float32')**: A method, used to build the model's structure. It accepts one argument `dtype`, indicating the data type, defaulting to 'float32'. This method creates all the convolutional layers, max pooling layers and fully connected layers that are needed for the model, and stores them in corresponding attributes.
- **fp(data, p)**: A method, used to perform forward propagation. It accepts two arguments `data` and `p`, indicating the input data and process number respectively. This method passes the input data through all the layers of the model and returns the output data.
- **loss(output, labels, p)**: A method, used to calculate the loss value. It accepts three arguments `output`, `labels` and `p`, indicating the output data, true labels and process number respectively. This method uses the loss function to calculate the difference between output data and true labels and returns the loss value.
- **GradientTape(data, labels, p):** A method, used to calculate the gradient value. It accepts three arguments data, labels and p, indicating the instance input data, true labels and device number respectively. This method uses a persistent gradient tape to record the operations and compute the gradient of the loss with respect to the parameters. This method returns the tape, output data and loss value.
- **opt(gradient, p)**: A method, used to perform optimization update. It accepts two arguments `gradient` and `p`, indicating the gradient value and process number respectively. This method uses the optimizer to update all the parameters of the model according to gradient value and returns updated parameters.
