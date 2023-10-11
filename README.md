Written by Dimitra Paranou, August 2023

Using deep learning and large protein language models to predict protein-membrane interfaces of peripheral membrane proteins 

The repository holds input files required to run the scripts for data preparation, feature extraction (embeddings of protein Language Models(pLM)), attention maps of pLMs investigation, and models training & Multi-layer Perceptron (MLP) fine tuning

The following files and directories/folders have been generated:

**Datasets/**
Contains the 2 datasets (DREAMM & PePr2DS) that we used to create our own, as well as our generated files with proteins (cluster results, aligned proteins, proteins with features (embeddings))

**models/**
Contains the 4 best fine-tuned MLP models trained on protTrans and ESM embeddings.

- 2 MLP models with 4 hidden layers (best_model_esm, best_model_protTrans)
- 2 MLP models with 1 hidden layer (best_model_esm_sl, best_model_protTrans_sl)

**scripts/**
Contains several project's scripts.

- attentions_lm.py -> Extract the attention maps from 2 pLMs and [1] Investigate if information about Interfacial Binding Sites (IBS) is encoded on pLMs hidden layers & [2] Train Logistic Regression models on attention maps and predict the IBS
- embeddings_lm.py -> Extract the embeddings of 2 pLMs so as to use them as features for the dataset
- explore_datasets.ipynb -> Explore and create the dataset files
- get_proteins.py -> Scirpt for downloading proteins from Uniprot / PDB
- ml_testing.py -> Testing of 4 Machine Learning (ML) classifiers - XGBoost, BalancedRandomForest, Single layer Perceptron, Multi layer Perceptron -
- mlp_tune.ipynb -> Fine-tuning of MLP using Bayesian Optimization

**tuning_trial/**
Contains the trial using Bayesian Optimization for fine-tuning the 4 MLP models

**visualization/**
Contains the files for creating visualization scripts (.pymol) and visualize the predictions of models (TP, FP, FN)
