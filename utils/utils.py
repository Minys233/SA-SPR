import requests
import hashlib
from tqdm import tqdm
import os
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
import numpy as np


def checksum(filename):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(url, filename, md5=None, keep=False):
    # check if the right file exists
    if os.path.isfile(filename) and checksum(filename) == md5:
        print(f"{filename} exists, MD5 hash check passed.")
        return filename
    # download it
    response = requests.get(url, stream=True)
    filesize = int(response.headers['content-length'])
    hash_md5 = hashlib.md5()
    with open(filename, 'wb') as fout:
        for chunk in tqdm(response.iter_content(4096), total=filesize//4096, ncols=80):
            fout.write(chunk)
            hash_md5.update(chunk)
    local_md5 = hash_md5.hexdigest()
    # check md5 hash of the file
    if md5 is not None:
        if local_md5 == md5:
            print(f"{filename}\tMD5 hash check passed.")
            return filename
        else:
            if keep:
                print(f"{filename}\tMD5 hash check NOT passed, but kept.")
                return filename
            else:
                print(f"{filename}\tMD5 hash check NOT passed, deleted.")
                os.remove(filename)
                return False
    else:
        print(f"{filename}\t not checking md5")
        return filename


def mol2vec_features(model, dataframe, smiles_col, target_col, pad_to):
    mollst = [Chem.MolFromSmiles(x) for x in dataframe[smiles_col]]
    sentences = [mol2alt_sentence(x, 1) for x in mollst]
    features = np.zeros([len(mollst), pad_to, model.vector_size])
    labels = np.reshape(np.array(dataframe[target_col]), (-1, 1))
    print("mean: ", labels.mean(), "std: ", labels.std())
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
    return features, labels

