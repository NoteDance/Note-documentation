class Sequential:
    def __init__(self):
        self.layer=[]  # List to store layers
        self.param=[]  # List to store parameters of the layers
        self.saved_data=[]  # List to store intermediate data if needed
        self.save_data_flag=[]  # List of flags indicating whether to save data for each layer
        self.use_data_flag=[]  # List of flags indicating whether to use saved data for each layer
        self.save_data_count=0  # Counter for the number of times data has been saved
        self.output_size=None  # Output size of the last layer
        self.train_flag=True  # Flag to indicate training mode

    def add(self,layer,save_data=False,use_data=False):
        # If the input layer is not a list
        if type(layer)!=list:
            if save_data==True:
                self.save_data_count+=1  # Increment save_data_count if save_data is True
            if use_data==True and hasattr(layer,'save_data_count'):
                layer.save_data_count=self.save_data_count  # Set save_data_count in the layer if use_data is True
            if use_data==True:
                self.save_data_count=0  # Reset save_data_count if use_data is True
            self.layer.append(layer)  # Add layer to the list without changes
            if hasattr(layer,'param'):
                self.param.extend(layer.param)  # Add layer's parameters to the param list
            if hasattr(layer,'output_size'):
                self.output_size=layer.output_size  # Update the output_size to the layer's output_size
            self.save_data_flag.append(save_data)  # Add save_data flag to the list
            self.use_data_flag.append(use_data)  # Add use_data flag to the list
        else:
            # If the input is a list of layers
            for layer in layer:
                self.layer.append(layer)  # Add layer to the list without changes
                if hasattr(layer,'param'):
                    self.param.extend(layer.param)  # Add layer's parameters to the param list
                if hasattr(layer,'output_size'):
                    self.output_size=layer.output_size  # Update the output_size to the layer's output_size
        return

    def __call__(self,data,train_flag=True):
        for i, layer in enumerate(self.layer):
            if not hasattr(layer,'train_flag'):
                if len(self.use_data_flag)==0 or self.use_data_flag[i]==False:
                    data = layer(data)  # Forward data through the layer
                else:
                    if hasattr(layer,'save_data_count'):
                        data=layer(self.saved_data)  # Use saved data if needed
                    else:
                        data=layer(data,self.saved_data.pop(0))  # Use saved data if needed
            else:
                if len(self.use_data_flag)==0 or self.use_data_flag[i]==False:
                    data = layer(data,train_flag)  # Forward data through the layer with train_flag
                else:
                    if hasattr(layer,'save_data_count'):
                        data=layer(self.saved_data, train_flag)  # Use saved data if needed with train_flag
                    else:
                        data=layer(data,self.saved_data.pop(0),train_flag)  # Use saved data if needed with train_flag
            if len(self.save_data_flag)>0 and self.save_data_flag[i]==True:
                self.saved_data.append(data)  # Save data if needed
        return data
