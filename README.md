Written by Dimitra Paranou, August 2023

Using deep learning and large protein language models to predict protein-membrane interfaces of peripheral membrane proteins

The repository holds input files required to run the scripts for data preparation, feature extraction (embeddings of protein Language Models(pLM)), attention maps of pLMs investigation, and models training & Multi-layer Perceptron (MLP) fine tuning

The following files and directories/folders have been generated:

**Datasets/**
Contains the 2 datasets (DREAMM & PePr2DS) that we used to create our own, as well as our generated files with proteins (cluster results, aligned proteins, proteins with features (embeddings)). The folder contains 2 subfolders with different dataset information.

- PDB_dataset -> contains the dataset with the PDB sequences
- Uniprot_dataset -> contains the dataset with the Uniprot sequences

**models/**
Contains the 4 best fine-tuned MLP models trained on protTrans and ESM embeddings.

- 2 MLP models with 4 hidden layers (best_model_esm, best_model_protTrans)
- 2 MLP models optimized with Optuna framework (best_model_esm_new, best_model_protTrans_new)

**scripts/**
Contains several project's scripts.

- attentions_lm.py -> Extract the attention maps from 2 pLMs and [1] Investigate if information about Interfacial Binding Sites (IBS) is encoded on pLMs hidden layers & [2] Train Logistic Regression models on attention maps and predict the IBS
- attentions.py -> Functions for extracting attention maps of proteins, create heatmaps for visualize the information in hidden layers of pLMs and train ML classifiers for predicting membrane-penetrating amino acids
- dataset-pdb.py -> All the functions for data preprocessing of dataset with PDB sequences, embeddings and attentions generation, fine-tuning and training of ML models
- embeddings_lm.py -> Extract the embeddings of 2 pLMs so as to use them as features for the Uniprot dataset
- explore_datasets.ipynb -> Explore and create the dataset files for Uniprot sequences
- get_proteins.py -> Scirpt for downloading proteins from Uniprot / PDB
- mlp_tune.ipynb -> Fine-tuning of MLP using Bayesian Optimization and keras tuner
- ml_tune.ipynb -> Fine-tuning of 5 ML classifiers - LGBM, XGBoost, BalancedRandomForest, Single layer Perceptron, Multi layer Perceptron - using Optuna library
- pairwise_seq_ident.py -> Create a file with the alignments of proteins
