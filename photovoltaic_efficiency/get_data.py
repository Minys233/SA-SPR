from utils import download
from pathlib import Path

if __name__ == '__main__':
    datadir = Path('./data')
    if not datadir.is_dir():
        datadir.mkdir(exist_ok=True)
    # TODO: remove redundant download
    # Mol2vec pre-trained model
    download(
        'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl',
        str(datadir/'mol2vec_model_300dim.pkl'),
        '943260d383420e9ff19168dc59cdc99e'
        )
    # PCE data
    download(
        'https://github.com/HIPS/neural-fingerprint/raw/master/data/2015-06-02-cep-pce/cep-processed.csv',
        str(datadir/'cep-processed.csv'),
        'b6d257ff416917e4e6baa5e1103f3929'
    )
