from pathlib import Path
from utils import download


if __name__ == '__main__':
    datadir = Path('./data')
    if not datadir.is_dir():
        datadir.mkdir(exist_ok=True)

    # Mol2vec pre-trained model
    download(
        'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl',
        str(datadir/'mol2vec_model_300dim.pkl'),
        '943260d383420e9ff19168dc59cdc99e'
        )

    # ESOL data
    download(
        'https://cloud.tsinghua.edu.cn/f/2cc3b125053a4275b6a2/?dl=1',
        str(datadir/'ESOL-solubility.csv'),
        'ac1580ec494ad7a0f6f040f9afce96cf'
        )
    download(
        'https://cloud.tsinghua.edu.cn/f/d3460ae6efc747a8802b/?dl=1',
        str(datadir/'ESOL-solubility-readme.txt'),
        'a0cfbfb4959ebf1f67b0685a5ef9fd9d'
        )
