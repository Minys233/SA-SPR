# SA-SPR
Reproduction: Identifying Structure−Property Relationships through SMILES Syntax Analysis with Self-Attention Mechanism

Since the source code is not provided in the original paper, and neithor does the author's github [repository](https://github.com/SYSU-RCDD/SA-SPR). Thus, this repository is here for some basic implementation and test for the self-attention mechanism.

Note: only 2 tasks of the paper is implemented, which are solubility prediction on ESOL and photovoltaic efficiency prediction on [NFP paper](https://github.com/HIPS/neural-fingerprint). I think other tasks are rather easier to implement provided the self-attention layer in this repository.


# Install
The required packaged shall be installed by mixing pip and conda. Refer to `environment.yml` or the error if your environmental requirements are not met.
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

# Repository structure
- utils
    - `layers.py`: the self-attention layer used in the paper.
    - `utils.py`: some util functions used to preprocess data mainly from [Mol2vec](https://github.com/samoturk/mol2vec) paper/model.
- solubility
    - `get_data.py`: download the training data.
    - `data.py`: load, split and preprocess the downloaded data.
    - `model.py`: RNN model construction.
    - `train-sa-bilstm.py`: train & basic evaluation on self-attention BiLSTM model (best in paper).
    - `train-simple-bilstm.py`: train & basic evaluation on simple BiLSTM model (control group in paper).
    - `*.csv`: log for grid-search and validation set MSE.
    - `*.ckpt`: pretrained TensorFlow checkpoints with best meta-parameters.
    - `*.png`: simple visualization with best trained models. (MSE in title is for validation set)
- photovoltaic_efficiency(*same as above*)
    
# How to run
If you want to train these models yourself, rather than directly check the result, pls follow the steps below.

Note: you may need 2 GPU card and over 1 day to complete the grid search. If your just train a model with certain configuration (maybe the best I tried), pls modify the `train-*-bilstm.py` scripts.
1. Clone this repo with `git clone`.
2. In the repo root, which should be `/some/path/SA-SPR`, run ``export PYTHONPATH=`pwd` ``.
    - this makes the import script in subdirectory correct.
3. From `solubility` or `photovoltaic_efficiency` subdir, run following scripts with no argument:
    - `get_data.py`: first download the data required.
    - `train-*-bilstm.py`: train & evaluation.

# Some explanation
1. For those who cannot load my pretrained model
    - Try load again on machines have GPU card
    - Provided results for 2 tasks, in pandas dataframe format.

1. In `photovoltaic-sa-bilstm-best.png` and `photovoltaic-simple-bilstm-best.png`, there are some points, which groundtruth are around 0, but prediction span widely. This is because I forget to filter out items with groundtruth 0 (invalid data) as [NFP paper](https://github.com/HIPS/neural-fingerprint) do. In raw data, they are strictly 0, but since I process them using mean and std value only from training set, so they are not strictly 0 in the result.
    - There are 821/29978(2.74%) such datapoints, nevermind.
    - After filtering, the MSE over whole dataset decrease to 0.716 (including training set, just for reference)


