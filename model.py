import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K


if tf.test.is_gpu_available():
    LSTM = keras.layers.CuDNNLSTM
else:
    LSTM = keras.layers.LSTM

# refer to https://github.com/thushv89/attention_keras
# and documentation of tf.keras.layers.Layer
# for how to define a custom layer.
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
        super(SelfAttentiveLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # ignore kwargs inorder to keep same signature with meta class
        # the input should have already been concatenated!
        if isinstance(inputs, list):
            raise TypeError('Argument `inputs` should have already been concatenated. '
                            'Do not directly pass the output of LSTM or GRU to this layer, '
                            'use `tf.concat` first to merge them into single tensor.'
                            'If your input sequences have variable length, please padd first.')
        # TODO: add mask for padded sequence
        H = inputs
        # input : (batch_size, n, vector_size)
        # 1. transpose to (batch_size, vector_size, n)
        H_trans = tf.transpose(inputs, perm=[0, 2, 1])
        # 2. matmul with W1: (self.da, vector_size) * (batch_size, vector_size, n) = (batch_size, self.da, n)
        # 3. tanh activation
        after_w1 = tf.tanh(self.W1 @ H_trans)  # matmul
        # 4. matmul with W2: (self.r, self.da) * (batch_size, self.da, n) = (batch_size, self.r, n)
        # 5. softmax activation
        # 6. the output matrix A will has a shape of (batch_size, self.r, n)
        A = tf.nn.softmax(self.W2 @ after_w1, axis=1)  # maybe wrong! which axis???
        # 7. matmul A with input H to get output Ma:
        # (batch_size, self.r, n) * (batch_size, n, vector_size) = (batch_size, self.r, vector_size)
        Ma = A @ H
        # NO BIAS
        return Ma


def build_simple_bilstm_model(pad_to, vector_size, lstm_hidden):
    inputs = keras.layers.Input(shape=(pad_to, vector_size), dtype='float32')
    sequence = keras.layers.Bidirectional(LSTM(lstm_hidden, return_sequences=True))(inputs)
    # sequence = keras.layers.LSTM(lstm_hidden, return_sequences=True)(inputs)
    sequence = keras.layers.Flatten()(sequence)
    # y = keras.layers.Dropout(0.5)(sequence)
    y = keras.layers.Dense(1)(sequence)
    # y = keras.layers.Activation(lambda x: tf.sign(x)*tf.sqrt(tf.abs(x)))(y)
    # y = keras.layers.Dense(1, activation='relu')(y)
    model = keras.models.Model(inputs=inputs, outputs=y)
    return model


def build_sa_bilstm_model(pad_to, vector_size, lstm_hidden, da, r):
    inputs = keras.layers.Input(shape=(pad_to, vector_size), dtype='float32')
    sequence = keras.layers.Bidirectional(LSTM(lstm_hidden, return_sequences=True))(inputs)
    # print(sequence)
    # concatenate = keras.layers.Concatenate()(*sequence)
    self_attention = SelfAttentiveLayer(da, r)(sequence)
    flatten = keras.layers.Flatten()(self_attention)
    y = keras.layers.Dense(1)(flatten)
    # may be activation could be added here
    model = keras.models.Model(inputs=inputs, outputs=y)
    return model


if __name__ == "__main__":
    model = build_simple_bilstm_model(40, 300, 150)
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mae'])
    model.summary()

    model = build_sa_bilstm_model(40, 300, 160, 100, 101)
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mae'])
    model.summary()