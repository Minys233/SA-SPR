import tensorflow as tf
from tensorflow import keras


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


if __name__ == "__main__":
    model = build_simple_bilstm_model(40, 300, 150)
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mae'])
    print(model.summary())