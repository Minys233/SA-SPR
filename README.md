# SA-SPR
Reproduction: Identifying Structure−Property Relationships through SMILES Syntax Analysis with Self-Attention Mechanism

# Install
The required packaged shall be installed by mixing pip and conda. Refer to `environment.yml` if your environmental requirements are not met.
- conda
    - rdkit
    - gensim
    - matplotlib
    - numpy
    - pandas
    - scikit-learn
    - scipy
    - tqdm
    - requests
    
- pip: 
    - tensorflow / tensorflow-gpu
    - mol2vec: via `pip install git+https://github.com/samoturk/mol2vec`

# Train & evaluate
from `SA-SPR`, which is the root dir, run ``export PYTHONPATH="${PYTHONPATH}:`pwd`"``. This makes import script works in subdirectories.

Please first make a new dir `data` in order to run `get-data.py` to download training data and mol2vec pretrained model.

Please then run `train-simple-bilstm.py` to train, grid search, evaluate the simple BiLSTM model. Feel free to modify the training script if you do not want to do grid search, refer to `solubility-simple-bilstm-summary.csv` to select hyper-parameters.

Note: you may change `LSTM` to `CuDNNLSTM` in `model.py` in order to train with a GPU.