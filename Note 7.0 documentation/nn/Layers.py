# Define a class that can store and apply different layers to data
class Layers:
    # Define the initialization method
    def __init__(self):
        # Initialize a list to store the layers
        self.layer=[]
        # Initialize a list to store the parameters of the layers
        self.param=[]
        # Initialize a list to store the intermediate data
        self.saved_data=[]
        # Initialize a list to store the flags of whether to save data or not
        self.save_data_flag=[]
        # Initialize a list to store the flags of whether to use data or not
        self.use_data_flag=[]
        # Initialize a counter to count the number of data to save
        self.save_data_count=0
        # Initialize a list to store the output sizes of the layers
        self.output_size_list=[]
        self.train_flag=True
    
    
    # Define a method to add a layer to the list
    def add(self,layer,save_data=False,use_data=False,axis=None):
        # If save_data is True, increment the save_data_count by 1
        if save_data==True:
            self.save_data_count+=1
        # If use_data is True and the layer has an attribute of save_data_count, assign it with the current save_data_count
        if use_data==True and hasattr(layer,'save_data_count'):
            layer.save_data_count=self.save_data_count
        # If use_data is True and the layer does not have an attribute of concat, clear the output_size_list
        if use_data==True and hasattr(layer,'concat')!=True:
            self.output_size_list=[]
        # If use_data is True, reset the save_data_count to 0
        if use_data==True:
            self.save_data_count=0
        # If the layer has an attribute of build, check if its input size is None and the current output size is not None
        if hasattr(layer,'build'):
            if layer.input_size==None and self.output_size!=None:
                # Set the layer's input size as the current output size
                layer.input_size=self.output_size
                # Build the layer with its input size
                layer.build()
                # Append the layer to the layer list
                self.layer.append(layer)
            else:
                # Append the layer to the layer list without building it
                self.layer.append(layer)
        else:
            # Append the layer to the layer list without building it
            self.layer.append(layer)
        # If the layer has an attribute of param, append it to the param list
        if hasattr(layer,'param'):
            self.param.append(layer.param)
        # If the layer has an attribute of output_size, set it as the current output size
        if hasattr(layer,'output_size'):
            self.output_size=layer.output_size
        # If the layer has an attribute of concat, calculate the new output size by adding up all the output sizes in the output_size_list according to its axis argument
        if hasattr(layer,'concat'):
            if layer.axis==-1 or layer.axis==2:
                self.output_size=self.output_size_list.pop(0)
                for i in range(1,layer.save_data_count):
                    self.output_size+=self.output_size_list.pop(0)
        # Append the save_data flag to the save_data_flag list
        self.save_data_flag.append(save_data)
        # Append the use_data flag to the use_data_flag list
        self.use_data_flag.append(use_data)
        # If save_data is True, append the current output size to the output_size_list
        if save_data==True:
            self.output_size_list.append(self.output_size)
        return
    
    
    # Define a method to apply all the layers in the list to an input data and return an output data
    def output(self,data,train_flag=True):
        # Loop through each layer in the list by index and value
        for i,layer in enumerate(self.layer):
            # If the layer has an attribute of output, check if it has an attribute of train_flag or not
            if hasattr(layer,'output'):
                if not hasattr(layer,'train_flag'):
                    # If it does not have an attribute of train_flag, check if use_data_flag is False or not for this index
                    if self.use_data_flag[i]==False:
                        # If use_data_flag is False, apply the layer's output method to data and assign it back to data 
                        data=layer.output(data)
                    else:
                        # If use_data_flag is True, check if the layer has an attribute of save_data_count or not 
                        if hasattr(layer,'save_data_count'):
                            # If it has an attribute of save_data_count, apply the layer's output method to the entire saved_data list and assign it back to data
                            data=layer.output(self.saved_data)
                        else:
                            # If it does not have an attribute of save_data_count, apply the layer's output method to data and the first element of the saved_data list and assign it back to data
                            data=layer.output(data,self.saved_data.pop(0))
                else:
                    # If it has an attribute of train_flag, check if use_data_flag is False or not for this index
                    if self.use_data_flag[i]==False:
                        # If use_data_flag is False, check if train_flag is False or not
                        if not train_flag:
                            # If train_flag is False, apply the layer's output method to data with train_flag as False and assign it back to data
                            data=layer.output(data,train_flag)
                        else:
                            # If train_flag is True, apply the layer's output method to data with train_flag as True and assign it back to data
                            data=layer.output(data)
                    else:
                        # If use_data_flag is True, check if the layer has an attribute of save_data_count or not 
                        if hasattr(layer,'save_data_count'):
                            # If it has an attribute of save_data_count, check if train_flag is False or not
                            if not train_flag:
                                # If train_flag is False, apply the layer's output method to the entire saved_data list with train_flag as False and assign it back to data
                                data=layer.output(self.saved_data,train_flag)
                            else:
                                # If train_flag is True, apply the layer's output method to the entire saved_data list with train_flag as True and assign it back to data
                                data=layer.output(self.saved_data)
                        else:
                            # If it does not have an attribute of save_data_count, check if train_flag is False or not
                            if not train_flag:
                                # If train_flag is False, apply the layer's output method to data and the first element of the saved_data list with train_flag as False and assign it back to data
                                data=layer.output(data,self.saved_data.pop(0),train_flag)
                            else:
                                # If train_flag is True, apply the layer's output method to data and the first element of the saved_data list with train_flag as True and assign it back to data
                                data=layer.output(data,self.saved_data.pop(0))
                # If save_data_flag is True for this index, append the current data to the saved_data list
                if self.save_data_flag[i]==True:
                    self.saved_data.append(data)
            elif not hasattr(layer,'concat'):
                # If the layer does not have an attribute of output and concat, check if use_data_flag is False or not for this index
                if self.use_data_flag[i]==False:
                    # If use_data_flag is False, apply the layer as a callable function to data and assign it back to data 
                    data=layer(data)
                else:
                    # If use_data_flag is True, apply the layer as a callable function to data and the first element of the saved_data list and assign it back to data 
                    data=layer(data,self.saved_data.pop(0))
                # If save_data_flag is True for this index, append the current data to the saved_data list
                if self.save_data_flag[i]==True:
                    self.saved_data.append(data)
            else:
                # If the layer has an attribute of concat, apply its concat method to the entire saved_data list and assign it back to data 
                data=layer.concat(self.saved_data)
                # If save_data_flag is True for this index, append the current data to the saved_data list
                if self.save_data_flag[i]==True:
                    self.saved_data.append(data)
        # Return the final output data
        return data
