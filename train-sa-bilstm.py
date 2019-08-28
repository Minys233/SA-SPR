import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
import numpy as np
from pathlib import Path

from data import load_ESOL
from model import build_sa_bilstm_model

import matplotlib.pyplot as plt


# almost same as train_simple_bilstm
def train_sa_bilstm(pad_to, lstm_hidden, da, r, lr, loss, savefigto):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = \
        load_ESOL('data/ESOL-solubility.csv', 'data/mol2vec_model_300dim.pkl', pad_to=pad_to)
    _, _, vector_size = train_x.shape
    model = build_sa_bilstm_model(pad_to=pad_to, vector_size=vector_size, lstm_hidden=lstm_hidden, da=da, r=r)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=loss, metrics=['mse'])
    print(model.summary())
    print(train_x.shape, train_y.shape)
    train_dataset = Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=128).batch(64,
                                                                                                  drop_remainder=True)
    val_dataset = Dataset.from_tensor_slices((val_x, val_y)).batch(32, drop_remainder=True)
    test_dataset = Dataset.from_tensor_slices((test_x, test_y)).batch(32, drop_remainder=True)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, update_freq='batch')
    earlystop_callback = keras.callbacks.EarlyStopping(patience=10)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'./checkpoints/model-sa-bilstm-{pad_to}-{lstm_hidden}-{da}-{r}-{lr}-{loss}.ckpt',
        save_best_only=True)
    model.fit(train_dataset,
              epochs=100,
              validation_data=val_dataset,
              callbacks=[tensorboard_callback, earlystop_callback, checkpoint_callback]
              )

    # std, mean
    predict = np.array(model.predict(test_x)).ravel() * 2.0965 - 3.058
    truth = np.array(test_y).ravel() * 2.0965 - 3.058

    plt.figure(figsize=(5, 5))
    plt.scatter(predict, truth)
    plt.plot([-8, 0], [-8, 0], 'r--')
    plt.axis([-8, 0, -8, 0])
    plt.xlabel("Prediction")
    plt.ylabel("Groundtruth")
    MSE = ((predict - truth) ** 2).mean()
    plt.title(f"MSE = {MSE:.3f}")
    plt.savefig(Path(savefigto) / f'./solubility_sa_bilstm-{pad_to}-{lstm_hidden}-{da}-{r}-{lr}-{loss}-{MSE:.4f}.png')


if __name__ == '__main__':
    # TODO: grid search, test, visialization
    pad_to_lst = [20, 40, 60, 80, 100]
    lstm_hidden_lst = [100, 150, 300]
    da_lst = [20, 40, 60, 80, 100, 200]
    r_lst = [10, 30, 50, 100, 150, 200]
    # lr_lst = [0.001, 0.0005, 0.0001]
    lr = 0.0001
    # loss_lst = ['mae', 'mse']
    loss = 'mse'
    savefigto = 'result'

    for pad_to in pad_to_lst:
        for lstm_hidden in lstm_hidden_lst:
            for da in da_lst:
                for r in r_lst:
                    train_sa_bilstm(pad_to, lstm_hidden, da, r, lr, loss, savefigto)
                    keras.backend.clear_session()

    with open('solubility-sa-bilstm-summary.csv', 'w') as fout:
        print('Max molecule size,LSTM hidden size,da,r,Learning rate,Loss function,MSE', file=fout)
        lines = []
        for fname in Path('result').glob('solubility_sa_bilstm*.png'):
            basename = fname.name.rsplit('.', 1)[0]
            _, pad_size, hidden_size, da, r, lr, loss, mse = basename.split('-')
            pad_size, hidden_size, da, r = int(pad_size), int(hidden_size), int(da), int(r)
            lr, mse = float(lr), float(mse)
            lines.append((pad_size, hidden_size, da, r, lr, loss, mse))
        lines = sorted(lines, key=lambda x: x[-1])
        for pad_size, hidden_size, da, r, lr, loss, mse in lines:
            print(f"{pad_size},{hidden_size},{da},{r},{lr},{loss},{mse:.4f}", file=fout)

    # dense input length: r*2hidden
