import requests
import hashlib
from tqdm import tqdm
import os


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


if __name__ == '__main__':
    # Mol2vec pre-trained model
    download(
        'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl',
        'data/mol2vec_model_300dim.pkl',
        '943260d383420e9ff19168dc59cdc99e'
        )

    # ESOL data
    download(
        'https://cloud.tsinghua.edu.cn/f/2cc3b125053a4275b6a2/?dl=1',
        'data/ESOL-solubility.csv',
        'ac1580ec494ad7a0f6f040f9afce96cf'
        )
    download(
        'https://cloud.tsinghua.edu.cn/f/d3460ae6efc747a8802b/?dl=1',
        'data/ESOL-solubility-readme.txt',
        'a0cfbfb4959ebf1f67b0685a5ef9fd9d'
        )
