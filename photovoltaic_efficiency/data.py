import pandas as pd
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from utils.utils import mol2vec_features

SEED = 20190827


def load_PCE(csv_path, mol2vec_path, pad_to=70):
    df = pd.read_csv(csv_path)
    model = word2vec.Word2Vec.load(mol2vec_path)
    features, labels = mol2vec_features(model, df, 'smiles', 'PCE', pad_to)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=SEED)
    features_train, features_val, labels_train, labels_val = \
        train_test_split(features_train, labels_train, test_size=0.25, random_state=SEED)
    # train: 0.8*0.75=0.6, val: 0.8*0.25=0.2, test: 0.2
    # this is better than min-max normalization
    mean, std = labels_train.mean(), labels_train.std()
    labels_train = (labels_train - mean) / std
    labels_val = (labels_val - mean) / std
    labels_test = (labels_test - mean) / std
    return (features_train, labels_train), (features_val, labels_val), (features_test, labels_test)


if __name__ == '__main__':
    # pad = 70 for maximum length of fingerprint of molecules.
    train, val, test = load_PCE('data/cep-processed.csv', 'data/mol2vec_model_300dim.pkl', pad_to=40)
    print(train[0].shape, train[1].shape)
    print(train[0].std(axis=2), train[1].std())
    # mean: 3.9005  std: 2.5375





