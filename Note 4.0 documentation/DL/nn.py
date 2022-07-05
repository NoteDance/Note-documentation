import tensorflow as tf


class nn:               #a simple example,a neural network class
    def __init__(self):
        self.weight1=tf.Variable(tf.random.normal([784,64]))
        self.bias1=tf.Variable(tf.random.normal([64]))
        self.weight2=tf.Variable(tf.random.normal([64,64]))
        self.bias2=tf.Variable(tf.random.normal([64]))
        self.weight3=tf.Variable(tf.random.normal([64,10]))
        self.bias3=tf.Variable(tf.random.normal([10]))
        self.param=[self.weight1,self.weight2,self.weight3,self.bias1,self.bias2,self.bias3]
        self.opt=tf.keras.optimizers.Adam()
        self.info='example'
    
    
    @tf.function
    def fp(self,data):
        with tf.device('GPU:0'):
            layer1=tf.nn.relu(tf.matmul(data,self.weight1)+self.bias1)
            layer2=tf.nn.relu(tf.matmul(layer1,self.weight2)+self.bias2)
            output=tf.matmul(layer2,self.weight3)+self.bias3
        return output
    
    
    def loss(self,output,labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=labels))        
