import tensorflow as tf
from Note.nn.layer.LSTM import LSTM
from Note.nn.layer.dense import dense
import Note.nn.process.optimizer as o


class lstm:
    def __init__(self):
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=o.Adam()
        self.bc=tf.Variable(0, dtype=tf.float64)
    

    def build(self):
        self.lstm1=LSTM(weight_shape=(3,50),timestep=10,return_sequence=True)
        self.lstm2=LSTM(weight_shape=(50,50),timestep=10)
        self.dense=dense([50, 10])
        self.param=[self.lstm1.weight_list,
                      self.lstm2.weight_list,
                      self.dense.weight,
                      self.dense.bias]
        return
    

    def fp(self,data):
        x=self.lstm1.output(data)
        x=self.lstm2.output(x)
        output=tf.matmul(x,self.dense.weight)+self.dense.bias
        return output
    
    
    def loss(self, output, labels):
        return self.loss_object(labels, output)
    
    
    def opt(self,gradient):
        param=self.optimizer.opt(gradient,self.param,self.bc)
        return param