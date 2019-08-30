import tensorflow as tf
from tensorflow.data import Dataset
config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.32
tf.keras.backend.set_session(tf.Session(config=config))
from tensorflow import keras
import numpy as np
from pathlib import Path

from photovoltaic_efficiency.data import load_PCE, Mol2vecLoader
from photovoltaic_efficiency.model import build_sa_bilstm_model

import matplotlib.pyplot as plt



# almost same as train_simple_bilstm
def train_sa_bilstm(pad_to, lstm_hidden, da, r, lr, loss, savefigto):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = \
        load_PCE('data/cep-processed.csv', 'data/mol2vec_model_300dim.pkl', pad_to=pad_to)
    _, _, vector_size = train_x.shape
    model = build_sa_bilstm_model(pad_to=pad_to, vector_size=vector_size, lstm_hidden=lstm_hidden, da=da, r=r)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=loss, metrics=['mse'])
    print(model.summary())
    print(train_x.shape, train_y.shape)
    train_dataset = Mol2vecLoader(train_x, train_y, pad_to, 128)
    val_dataset = Mol2vecLoader(val_x, val_y, pad_to, 32)
    test_dataset = Mol2vecLoader(test_x, test_y, pad_to, 32)
    # This eats huge HD space!
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, update_freq='batch')
    earlystop_callback = keras.callbacks.EarlyStopping(patience=10)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'./checkpoints/model-sa-bilstm-{pad_to}-{lstm_hidden}-{da}-{r}-{lr}-{loss}.ckpt',
        save_best_only=True)
    model.fit_generator(train_dataset,
                        epochs=1000,
                        validation_data=val_dataset,
                        callbacks=[earlystop_callback, checkpoint_callback]
                        )

    # std, mean
    predict = np.array(model.predict(test_x)).ravel() * 2.0965 + 3.9005
    truth = np.array(test_y).ravel() * 2.0965 + 3.9005

    plt.figure(figsize=(5, 5))
    plt.scatter(predict, truth, marker='.', c='b')
    plt.plot([-8, 12], [-8, 12], 'r--')
    plt.axis([-8, 12, -8, 12])
    plt.xlabel("Prediction")
    plt.ylabel("Groundtruth")
    MSE = ((predict - truth) ** 2).mean()
    plt.title(f"MSE = {MSE:.3f}")
    plt.savefig(Path(savefigto) / f'./photovoltaic_sa_bilstm-{pad_to}-{lstm_hidden}-{da}-{r}-{lr}-{loss}-{MSE:.4f}.png')
    plt.close()


if __name__ == '__main__':
    import sys
    arg = int(sys.argv[1])
    pad_to_lst = [20, 40, 60, 70]
    lstm_hidden_lst = [100, 150, 300, 450]
    da_lst = [20, 40, 60, 80, 100, 200]
    if arg == 1:
        r_lst = [5, 10, 15] 
    elif arg == 2:
        r_lst = [20, 30, 50]
    # lr_lst = [0.001, 0.0005, 0.0001]
    lr = 0.0001
    # loss_lst = ['mae', 'mse']
    loss = 'mse'
    savefigto = 'result'
    Path(savefigto).mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    with tf.device('gpu:0'):
        for pad_to in pad_to_lst:
            for lstm_hidden in lstm_hidden_lst:
                for da in da_lst:
                    for r in r_lst:
                        train_sa_bilstm(pad_to, lstm_hidden, da, r, lr, loss, savefigto)
                        keras.backend.clear_session()

    with open('photovoltaic-sa-bilstm-summary.csv', 'w') as fout:
        print('Max molecule size,LSTM hidden size,da,r,Learning rate,Loss function,MSE', file=fout)
        lines = []
        for fname in Path(savefigto).glob('photovoltaic_sa_bilstm*.png'):
            basename = fname.name.rsplit('.', 1)[0]
            _, pad_size, hidden_size, da, r, lr, loss, mse = basename.split('-')
            pad_size, hidden_size, da, r = int(pad_size), int(hidden_size), int(da), int(r)
            lr, mse = float(lr), float(mse)
            lines.append((pad_size, hidden_size, da, r, lr, loss, mse))
        lines = sorted(lines, key=lambda x: x[-1])
        for pad_size, hidden_size, da, r, lr, loss, mse in lines:
            print(f"{pad_size},{hidden_size},{da},{r},{lr},{loss},{mse:.4f}", file=fout)

    # dense input length: r*2hidden
