import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

# refer to https://github.com/thushv89/attention_keras
# as an example how to define a custom layer.
class SelfAttentiveLayer(keras.layers.Layer):
    def __init__(self, da, r, **kwargs):
        super(SelfAttentiveLayer, self).__init__(**kwargs)
        self.da = da
        self.r = r

    def build(self, input_shape):
        # concat of outputs of BiLSTM, with shape (batch_size, n, vector_size)
        assert isinstance(input_shape, tf.TensorShape)
        n, vector_size = input_shape[1], input_shape[2]
        self.W1 = self.add_weight(name='W1_attention',
                                  shape=tf.TensorShape((self.da, vector_size)),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2_attention',
                                  shape=tf.TensorShape((self.r, self.da)),
                                  initializer='glorot_uniform',
                                  trainable=True)

        # input : (batch_size, n, vector_size)
        # 1. transpose to (batch_size, vector_size, n)
        # 2. matmul with W1: (self.da, vector_size) * (batch_size, vector_size, n) = (batch_size, self.da, n)
        # 3. tanh activation
        # 4. matmul with W2: (self.r, self.da) * (batch_size, self.da, n) = (batch_size, self.r, n)
        # 5. the output Ma will has a shape of (batch_size, self.r, n)
        # NO BIAS


def build_simple_bilstm_model(pad_to, vector_size, lstm_hidden):
    inputs = keras.layers.Input(shape=(pad_to, vector_size), dtype='float32')
    sequence = keras.layers.Bidirectional(keras.layers.LSTM(lstm_hidden, return_sequences=True))(inputs)
    # sequence = keras.layers.LSTM(lstm_hidden, return_sequences=True)(inputs)
    sequence = keras.layers.Flatten()(sequence)
    # y = keras.layers.Dropout(0.5)(sequence)
    y = keras.layers.Dense(1)(sequence)
    # y = keras.layers.Activation(lambda x: tf.sign(x)*tf.sqrt(tf.abs(x)))(y)
    # y = keras.layers.Dense(1, activation='relu')(y)
    model = keras.models.Model(inputs=inputs, outputs=y)
    return model


def build_sa_bilstm_model(pad_to, vector_size, lstm_hidden):
    inputs = keras.layers.Input(shape=(pad_to, vector_size), dtype='float32')


if __name__ == "__main__":
    model = build_simple_bilstm_model(40, 300, 150)
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mae'])
    print(model.summary())