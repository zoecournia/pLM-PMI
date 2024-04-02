Written by Dimitra Paranou, August 2023

Using deep learning and large protein language models to predict protein-membrane interfaces of peripheral membrane proteins

The repository holds input files required to run the scripts for data preparation, feature extraction (embeddings of protein Language Models(pLM)) and models training & fine-tuning

The following files and directories/folders have been generated:

**Datasets/**
Contains the 2 datasets (DREAMM & PePr2DS) that we used to create our own, as well as our generated files with proteins (cluster results, aligned proteins, proteins with features (embeddings)). The folder contains 2 subfolders with different dataset information.

- PDB_dataset -> contains the dataset with the PDB sequences
- Uniprot_dataset -> contains the dataset with the Uniprot sequences

**models/**
Contains the 2 best fine-tuned MLP models trained on protTrans and ESM embeddings in Uniprot sequences and the 2 best fine-tuned LGBM models trained on protTrans and ESM embeddings in PDB sequences.

- 2 MLP models optimized with Optuna framework (best_model_esm, best_model_protTrans) - for Uniprot dataset
- 2 LGBM models trianed on PDB datasets optimized with Optuna framework - for PDB dataset

**scripts/**
Contains several project's scripts.

- dataset-pdb.py -> All the functions for data preprocessing of dataset with PDB sequences, embeddings generation, fine-tuning and training of ML models, predict new proteins
- embeddings_lm.py -> Extract the embeddings of 2 pLMs so as to use them as features for the Uniprot dataset
- explore_datasets.ipynb -> Explore and create the dataset files for Uniprot sequences
- get_proteins.py -> Scirpt for downloading proteins from Uniprot / PDB
- ml_tune.ipynb -> Fine-tuning of 5 ML classifiers - LGBM, XGBoost, BalancedRandomForest, Single layer Perceptron, Multi layer Perceptron - using Optuna library
- pairwise_seq_ident.py -> Create a file with the alignments of proteins

### How to run the scripts

First of all, you will need `Python >= 3.9` and a conda environment to install all the appropriate libraries by running:

```
# 1. activate the environment
conda activate my_env

# 2. install all the depedences
pip install -r requirements.txt

# 3. run a script
python scripts/dataset-pdb.py
```

For the case of Uniprot sequences, to predict membrane interacting amino acids for new proteins using the MLP models:

1. Edit the _proteins.json file_ in the **extra_proteins** folder where you set the Uniprot code and the matching PDB code and chain id for the visualization.
2. In the **examine_new_proteins.py** set the _model_name_ to be 'esm' or 'protTrans' that defines the language model that you want to use.

For the case of PDB sequences, to predict membrane interacting amino acids for new proteins using the LGBM models:

1. Edit the _mode_ to be 'predict_new_proteins', set the *mlp_model* with the desire pLM model ('esm' or 'protTrans'), and add the PDB codes in the variable *new_proteins* that you want to examine in the **dataset-pdb.py** script.
