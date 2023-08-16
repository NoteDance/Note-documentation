import tensorflow as tf


class lstm:
    def __init__(self,encoder):
        self.model=tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
        ])
        self.param=self.model.weights[1:]
        self.opt=tf.keras.optimizers.Adam()
    
    
    def fp(self,data):
        output=self.model(data)
        return output
    
    
    def loss(self,output,labels):
        bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce(labels,output)