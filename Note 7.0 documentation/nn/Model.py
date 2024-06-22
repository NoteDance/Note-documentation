import tensorflow as tf
from Note import nn

# Define a class Model to encapsulate neural network components and operations.
class Model:
    # Class variables for storing parameters and layer information.
    param = []
    param_dict = dict()
    param_dict['dense_weight'] = []
    param_dict['dense_bias'] = []
    param_dict['conv2d_weight'] = []
    param_dict['conv2d_bias'] = []
    layer_dict = dict()
    layer_param = dict()
    layer_list = []
    layer_eval = dict()
    counter = 0
    name_list = []
    ctl_list = []
    ctsl_list = []
    name = None
    name_ = None
    train_flag = True
    
    # Initialize the Model object and copy class variables to instance variables.
    def __init__(self):
        Model.init()
        self.param = Model.param
        self.param_dict = Model.param_dict
        self.layer_dict = Model.layer_dict
        self.layer_param = Model.layer_param
        self.layer_list = Model.layer_list
        self.layer_eval = Model.layer_eval
        self.head = None
        self.head_ = None
        self.ft_flag = 0
        self.detach_flag = False
        
    # Class method to increment the counter and append a new layer name.
    def add():
        Model.counter += 1
        Model.name_list.append('layer' + str(Model.counter))
    
    # Class method to apply a function to all layers in the current scope.
    def apply(func):
        for layer in Model.layer_dict[Model.name]:
            func(layer)
        if len(Model.name_list) > 0:
            Model.name_list.pop()
            if len(Model.name_list) == 0:
                Model.name = None
    
    # Detach the current instance from the class variables, creating a copy.
    def detach(self):
        if self.detach_flag:
            return
        self.param = Model.param.copy()
        self.param_dict = Model.param_dict.copy()
        self.layer_dict = Model.layer_dict.copy()
        self.layer_param = Model.layer_param.copy()
        self.layer_list = Model.layer_list.copy()
        self.layer_eval = Model.layer_eval.copy()
        self.detach_flag = True
    
    # Set the training flag for the model and its layers.
    def training(self, flag=False):
        Model.train_flag = flag
        for layer in self.layer_list:
            if hasattr(layer, 'train_flag'):
                layer.train_flag = flag
            else:
                layer.training = flag
    
    # Add a dense layer to the model.
    def dense(self, num_classes, dim, weight_initializer='Xavier', use_bias=True):
        self.head = nn.dense(num_classes, dim, weight_initializer, use_bias=use_bias)
        return self.head
    
    # Add a 2D convolutional layer to the model.
    def conv2d(self, num_classes, dim, kernel_size=1, weight_initializer='Xavier', padding='SAME', use_bias=True):
        self.head = nn.conv2d(num_classes, kernel_size, dim, weight_initializer=weight_initializer, padding=padding, use_bias=use_bias)
        return self.head
    
    # Method for fine-tuning the model by replacing the head layer.
    def fine_tuning(self, num_classes, flag=0):
        self.ft_flag = flag
        if flag == 0:
            self.head_ = self.head
            if isinstance(self.head, nn.dense):
                self.head = nn.dense(num_classes, self.head.input_size, self.head.weight_initializer, use_bias=self.head.use_bias)
            elif isinstance(self.head, nn.conv2d):
                self.head = nn.conv2d(num_classes, self.head.kernel_size, self.head.input_size, weight_initializer=self.head.weight_initializer, padding=self.head.padding, use_bias=self.head.use_bias)
            self.param[-len(self.head.param):] = self.head.param
            for param in self.param[:-len(self.head.param)]:
                param._trainable = False
        elif flag == 1:
            for param in self.param[:-len(self.head.param)]:
                param._trainable = True
        else:
            self.head, self.head_ = self.head_, self.head
            self.param[-len(self.head.param):] = self.head.param
            for param in self.param[:-len(self.head.param)]:
                param._trainable = True
    
    # Apply weight decay to parameters.
    def apply_decay(self, str, weight_decay, flag=True):
        if flag == True:
            for param in self.param_dict[str]:
                param.assign(weight_decay * param)
        else:
            for param in self.param_dict[str]:
                param.assign(param / weight_decay)
    
    # Cast the parameters to a specified data type.
    def cast_param(self, key, dtype):
        for param in self.param_dict[key]:
            param.assign(tf.cast(param, dtype))
    
    # Freeze the parameters of a specific layer.
    def freeze(self, name):
        for param in self.layer_param[name]:
            param._trainable = False
    
    # Unfreeze the parameters of a specific layer.
    def unfreeze(self, name):
        for param in self.layer_param[name]:
            param._trainable = True
    
    # Set the evaluation mode for layers.
    def eval(self, name=None, flag=True):
        if flag:
            for layer in self.layer_eval[name]:
                layer.train_flag = False
        else:
            for name in self.layer_eval.keys():
                for layer in self.layer_eval[name]:
                    layer.train_flag = True
    
    # Convert shared lists to a list.
    def convert_to_list():
        for ctl in Model.ctl_list:
            ctl()
    
    # Convert lists to a shared list using a manager.
    def convert_to_shared_list(manager):
        for ctsl in Model.ctsl_list:
            ctsl(manager)
    

    def init():
        Model.param.clear()
        Model.param_dict['dense_weight'].clear()
        Model.param_dict['dense_bias'].clear()
        Model.param_dict['conv2d_weight'].clear()
        Model.param_dict['conv2d_bias'].clear()
        Model.layer_dict.clear()
        Model.layer_param.clear()
        Model.layer_list.clear()
        Model.layer_eval.clear()
        Model.counter = 0
        Model.name_list = []
        Model.ctl_list.clear()
        Model.ctsl_list.clear()
        Model.name = None
        Model.name_ = None
        Model.train_flag = True
