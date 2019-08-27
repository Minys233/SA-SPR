from rdkit import Chem
import pandas as pd
import numpy as np
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
from sklearn.model_selection import train_test_split

SEED = 20190827


def load_ESOL(csv_path, mol2vec_path, pad_to=40):
    df = pd.read_csv(csv_path)
    df['RDKitMol'] = [Chem.MolFromSmiles(x) for x in df['SMILES']]
    model = word2vec.Word2Vec.load(mol2vec_path)
    sentences = [mol2alt_sentence(x, 1) for x in df['RDKitMol']]
    features = np.zeros([len(df), pad_to, model.vector_size])
    labels = np.reshape(np.array(df['measured log(solubility:mol/L)']), (-1, 1))
    print("mean: ", labels.mean(), "std: ",labels.std())
    for idx, sentence in enumerate(sentences):
        count = 0
        for word in sentence:
            if count == pad_to:
                break
            try:
                features[idx, count] = model.wv[word]
                count += 1
            except KeyError as e:
                pass
    assert features.shape[0] == labels.shape[0]
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=SEED)
    features_train, features_val, labels_train, labels_val = \
        train_test_split(features_train, labels_train, test_size=0.2, random_state=SEED)
    # train: 0.8*0.8=0.64, val: 0.8*0.2=0.16, test: 0.2
    # this is better than min-max normalization
    mean, std = labels_train.mean(), labels_train.std()
    labels_train = (labels_train - mean) / std
    labels_val = (labels_val - mean) / std
    labels_test = (labels_test - mean) / std
    return (features_train, labels_train), (features_val, labels_val), (features_test, labels_test)


if __name__ == '__main__':
    train, val, test = load_ESOL('data/ESOL-solubility.csv', 'data/mol2vec_model_300dim.pkl', pad_to=40)
    print(train[0].shape, train[1].shape)
    print(train[0].std(axis=2), train[1].std())
    # mean: -3.058  std: 2.0965
    # min: -11.6    max: 1.58





