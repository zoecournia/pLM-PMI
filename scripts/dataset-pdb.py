import pandas as pd
import numpy as np
import json
from Bio.PDB import *
from tqdm import tqdm
import os
import optuna
import joblib

from transformers import T5Tokenizer, T5EncoderModel, EsmTokenizer, EsmModel
import torch
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

has_gpu = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
project_path = '/home/dimitra/master_thesis'

amino_acids_3l_key = {"ARG":"R", "HIS":"H", "LYS":"K", "ASP":"D", "GLU":"E", "SER":"S", "THR":"T", "ASN":"N", "GLN": "Q", "CYS":"C", "SEC": "U","GLY":"G","PRO": "P", "ALA":"A","VAL":"V","ILE":"I", "LEU": "L", "MET":"M", "MSE": "M", "PHE": "F", "TYR": "Y", "TRP":"W"}
amino_acids_1l_key = {'R': 'ARG', 'H': 'HIS', 'K': 'LYS', 'D': 'ASP', 'E': 'GLU', 'S': 'SER', 'T': 'THR', 'N': 'ASN', 'Q': 'GLN', 'C': 'CYS', 'U': 'SEC', 'G': 'GLY', 'P': 'PRO', 'A': 'ALA', 'V': 'VAL', 'I': 'ILE', 'L': 'LEU', 'M': 'MET', 'F': 'PHE', 'Y': 'TYR', 'W': 'TRP'}
amino_acids = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'SEC', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']


"""
Mode -> Functionalities of script

1. create_pdb_labeling
2. create_fasta_file
3. filter_proteins
4. generate_embeddings
5. train_algorithm
6. predict_new_proteins
"""
mode = 'predict_new_proteins'
mlp_model = 'protTrans' # esm, protTrans
dataset_type = 'all' # all, dreamm

# Define the PDB codes of the new proteins for which we want to predict the membrane binding residues
new_proteins = ['5EDM']

"""
Read the PePrd2DS dataset and appropriatly format the data
"""
def read_peprd2ds(file_path):
  pepr_dataset = pd.read_csv(file_path)

  # rename IBS with label and place the column at the end
  pepr_dataset = pepr_dataset.rename(columns={'IBS':'label'})
  pepr_dataset = pepr_dataset[['pdb', 'chain_id', 'residue_name', 'residue_number', 'label']]

  # if there is no IBS value for a pdb and chain_id, remove the pdb with the corresponding chain_id
  pepr_dataset = pepr_dataset.groupby(['pdb', 'chain_id']).filter(lambda x: x['label'].any())
  pepr_dataset = pepr_dataset.reset_index(drop=True)

  # for each pdb and chain_id, short the residues by residue_number
  pepr_dataset = pepr_dataset.sort_values(['pdb', 'chain_id', 'residue_number']).reset_index(drop=True)

  return pepr_dataset

"""
Read the DREAMM dataset and appropriatly format the data
"""
def read_dreamm(file_path):
  # read the json file
  with open(file_path) as json_file:
      dreamm = json.load(json_file)

  # create a dataframe from the json file with dreamm labeling
  dreamm_df = pd.DataFrame()
  for pdb in dreamm:
      for residue in dreamm[pdb]:
          # check if pdb has >4 characters - case of different chains ex. 1a2y_A, 
          if len(pdb) > 4:
              new_row_df = pd.DataFrame({'pdb': pdb[:4].upper(), 'chain_id': pdb[-1], 'residue_name': amino_acids_1l_key[residue[0]], 'residue_number': int(residue[1:]), 'label': True}, index=[0])
              dreamm_df = pd.concat([dreamm_df, new_row_df], ignore_index=True)
          else:
              new_row_df = pd.DataFrame({'pdb': pdb.upper(), 'chain_id': 'A', 'residue_name': amino_acids_1l_key[residue[0]], 'residue_number': int(residue[1:]), 'label': True}, index=[0])
              dreamm_df = pd.concat([dreamm_df, new_row_df], ignore_index=True)

  return dreamm_df

"""
Create dataframe with sequences of new proteins
"""
def get_proteins_sequences(pdb_list):
  dataset = pd.DataFrame()

  for pdb in pdb_list:
      pdbl = PDBList()
      pdbl.retrieve_pdb_file(pdb, pdir = 'pdbs/', file_format = 'pdb')

      parser = PDBParser(PERMISSIVE = True, QUIET = True) 
      data = parser.get_structure(pdb.lower(),"pdbs/pdb"+pdb.lower()+".ent")

      models = list(data.get_models())
      chains = list(models[0].get_chains()) 
      for chain in chains:
          residues = list(chain.get_residues())

          for residue in residues:
              # check that the residue is an amino acid and if the residue number and residue name are in the merged dataset - if not, add it
              if residue.get_resname() in amino_acids :
                  new_row_df = pd.DataFrame({'pdb': pdb, 'chain_id': chain.get_id(), 'residue_name': residue.get_resname(), 'residue_number': residue.get_id()[1], 'label': False}, index=[0])
                  dataset = pd.concat([dataset, new_row_df], ignore_index=True)

  # sort the dataset by pdb, chain_id and residue_number
  dataset = dataset.sort_values(['pdb', 'chain_id', 'residue_number']).reset_index(drop=True)

  return dataset

"""
Read dataset from csv file
"""
def load_dataset(file_path, index_col=None):
  return pd.read_csv(file_path, dtype={"pdb": str}, index_col=index_col)

def load_ml_model(path):
  # load the model from disk
  loaded_model = joblib.load(path)
  return loaded_model

"""
Concatenate the PePrd2DS and DREAMM datasets
"""
def concatenate_datasets(pepr_dataset, dreamm_df):
  # Check if there are same pdbs in both datasets
  same_pdbs = dreamm_df[dreamm_df['pdb'].isin(pepr_dataset['pdb'])]

  # Find pdbs of dreamm that are not in pepr
  pdbs_not_in_pepr = dreamm_df[~dreamm_df['pdb'].isin(pepr_dataset['pdb'])]

  # Merge the two datasets
  merged_dataset = pd.concat([pepr_dataset, dreamm_df], ignore_index=True)

  # remove duplicates with same residue_name, residue_number of a specific pdb and chain_id
  merged_dataset = merged_dataset.drop_duplicates(subset=['pdb', 'chain_id', 'residue_name', 'residue_number'], keep='first')

  # sort the dataset by pdb, chain_id and residue_number
  merged_dataset = merged_dataset.sort_values(['pdb', 'chain_id', 'residue_number']).reset_index(drop=True)

  return merged_dataset

"""
Add residues that are not in the dataset with label False 
- The pdbs from DREAMM have residues that are only True membrane binding residues
"""
def add_false_labels(dataset):
    # get the pdbs that has only True labels and add the residues with label False
    pdb_with_all_true_labels = dataset.groupby(['pdb', 'chain_id']).apply(lambda group: all(group['label'])).reset_index(name='all_true_labels')
    result_pdb_list = pdb_with_all_true_labels[pdb_with_all_true_labels['all_true_labels']]['pdb'].tolist()
    
    for pdb in result_pdb_list:
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb, pdir = 'pdbs/', file_format = 'pdb')

        parser = PDBParser(PERMISSIVE = True, QUIET = True) 
        data = parser.get_structure(pdb.lower(),"pdbs/pdb"+pdb.lower()+".ent")

        models = list(data.get_models())
        chains = list(models[0].get_chains()) 
        for chain in chains:
            residues = list(chain.get_residues())

            # check if the chain exist in dataset - otherwise skip the chain
            if dataset[(dataset['pdb'] == pdb) & (dataset['chain_id'] == chain.get_id())].empty:
                continue
            
            for residue in residues:
                # check that the residue is an amino acid and if the residue number and residue name are in the merged dataset - if not, add it
                if residue.get_resname() in amino_acids and \
                dataset[(dataset['pdb'] == pdb) & 
                              (dataset['chain_id'] == chain.get_id()) & 
                              (dataset['residue_number'] == residue.get_id()[1]) & 
                              (dataset['residue_name'] == residue.get_resname())].empty:
                    new_row_df = pd.DataFrame({'pdb': pdb, 'chain_id': chain.get_id(), 'residue_name': residue.get_resname(), 'residue_number': residue.get_id()[1], 'label': False}, index=[0])
                    dataset = pd.concat([dataset, new_row_df], ignore_index=True)

    # sort the dataset by pdb, chain_id and residue_number
    dataset = dataset.sort_values(['pdb', 'chain_id', 'residue_number']).reset_index(drop=True)

    return dataset

"""
Create the fasta file for the dataset filtering using the CDHIT tool
"""
def create_fasta_file(df, file_name):
    file = open(project_path + file_name, "a")

    for protein in df.pdb.unique():
        protein_df = df[df.pdb == protein]
        chains = protein_df.chain_id.unique()

        for chain in chains:
            protein_df_chain = protein_df[protein_df.chain_id == chain]

            file.write('>' + protein + '_' + chain + '\n')
            file.write(''.join(protein_df_chain.residue_1l))
            file.write('\n\n')

    file.close()

"""
Keep only the representative pdb from the cdhit results
"""
def keep_representative_pdb(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # keep only the representative pdb
    filtered_pdb = [line.split()[2][1:-3] for line in lines if '*' in line]

    # from the pdbs make the first 4 characters keys and the last character values in a dictionary
    filtered_pdb = {pdb[:4]: pdb[-1] for pdb in filtered_pdb}

    return filtered_pdb
   
"""
Write the dataset to a csv file
"""
def save_dataset(merged_dataset, file_path):
    merged_dataset.to_csv(file_path,  index=False)
    
def load_model(mlp_model):
  if mlp_model == 'protTrans':
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)
    model = model.to(torch.float32)
  else:
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False)
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

  # load model to GPU
  model = model.to(device)

  # Get the number of embeddings
  num_hidden_size = model.config.hidden_size
  # Get the number of hidden layers and attention heads
  num_hidden_layers = model.config.num_hidden_layers
  num_attention_heads = model.config.num_attention_heads

  print('Model successfully loaded and running in',device)

  return model, tokenizer, num_hidden_size, num_hidden_layers, num_attention_heads

def generate_embeddings(df, model, tokenizer):
    results = {}
    # for each pdb and chain_id, get the residue_1l values and join them without a space
    df['sequence'] = df.groupby(['pdb', 'chain_id'])['residue_1l'].transform(lambda x: ''.join(x))

    # repl. all non-standard AAs and map them to unknown/X
    proteins_with_seq = df[['pdb', 'sequence']].groupby('pdb').apply(lambda x: list(np.unique(x['sequence']))[0].replace('U','X').replace('Z','X').replace('O','X')).to_dict()

    seq_dict = sorted(proteins_with_seq.items(), key=lambda kv: len( proteins_with_seq[kv[0]] ), reverse=True )

    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1)):
      seq = seq
      seq_len = len(seq)

      # get also the chain_id of the specific sequence, by matching the pdb_id and the sequence
      chain_id = df[(df['pdb'] == pdb_id) & (df['sequence'] == seq)].chain_id.unique()[0]

      if mlp_model == 'protTrans':
        seq = ' '.join(list(seq)) # add space between each AA for protTrans model

      # Tokenize the sequence
      token_encoding = tokenizer(seq, return_tensors="pt", add_special_tokens=False).to(device)

      try:
        with torch.no_grad():
          embedding_repr = model(**token_encoding)

      except RuntimeError:
        print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
        continue
      
      # Save the embeddings
      results[pdb_id] = {}
      results[pdb_id][chain_id] = embedding_repr.last_hidden_state.detach().cpu().numpy().squeeze()

    return results

def create_embeddings_dataset(df, embeddings, num_hidden_size):
    if 'label' in df.columns:
      all_columns = ['pdb', 'chain_id', 'residue_1l', 'residue_number'] + [f'Feature_{i+1}' for i in range(num_hidden_size)] + ['label']
    else:
      all_columns = ['pdb', 'chain_id', 'residue_1l', 'residue_number'] + [f'Feature_{i+1}' for i in range(num_hidden_size)]
    
    df_embeddings = pd.DataFrame(columns=all_columns)

    for protein in tqdm(embeddings.keys()):
        chain = list(embeddings[protein].keys())[0]
        
        if 'label' in df.columns:
         temp_df = pd.DataFrame(np.array(embeddings[protein][chain]), columns=all_columns[4:-1])  # Feature columns only
        else:
         temp_df = pd.DataFrame(np.array(embeddings[protein][chain]), columns=all_columns[4:])  # Feature columns only
        
        temp_df.insert(0, 'pdb', protein)
        temp_df.insert(1, 'chain_id', df[(df.pdb == protein) & (df.chain_id == chain)].chain_id.values)
        temp_df.insert(2, 'residue_1l', df[(df.pdb == protein) & (df.chain_id == chain)].residue_1l.values)
        temp_df.insert(3, 'residue_number', df[(df.pdb == protein) & (df.chain_id == chain)].residue_number.values)
        
        if 'label' in df.columns:
          temp_df['label'] = df[(df.pdb == protein) & (df.chain_id == chain)].label.values.astype(bool)

        # Use append for potentially better performance with many concatenations
        df_embeddings = pd.concat([df_embeddings, temp_df], ignore_index=True)
    
    return df_embeddings

def generate_pymols(df, is_extra_proteins=False):
    # create folder if not exists for saving the results
    if is_extra_proteins:
       if not os.path.exists(project_path + '/visualization_scripts/extra_test_proteins_pdb_' + mlp_model):
          os.mkdir(project_path + '/visualization_scripts/extra_test_proteins_pdb_' + mlp_model)
    else:
      if not os.path.exists(project_path + '/visualization_scripts/test_proteins_dreamm_pdb_' + mlp_model):
          os.mkdir(project_path + '/visualization_scripts/test_proteins_pdb_' + mlp_model)
    
    for protein in df.pdb.unique():
      protein_df = df[df.pdb == protein]
      chain = protein_df.chain_id.unique()[0]

      if (protein_df.predicted == 0).all():
         print("All predictions are 0 " + protein)
      
      TP = 'select TP, '
      FP = 'select FP, '
      FN = 'select FN, '

      if is_extra_proteins:
        for index, row in protein_df.iterrows():
            if row.predicted == 1:
              TP += f'chain {chain} and resid {row.residue_number} or '
      else:
        # Create a boolean mask for rows where label and predicted are different or both are 1
        mask = (protein_df['label'] != protein_df['predicted']) | (protein_df['label'] == 1)

        # iterrate through 
        for index, row in protein_df[mask].iterrows():
            if row.label == 1 and row.predicted == 1:
              TP += f'chain {chain} and resid {row.residue_number} or '
            elif row.label == 0 and row.predicted == 1:
              FP += f'chain {chain} and resid {row.residue_number} or '
            elif row.label == 1 and row.predicted == 0:
              FN += f'chain {chain} and resid {row.residue_number} or '
            else:
              print("something went wrong")
      
      # create pymol script
      with open(project_path + '/scripts/pymol_template.txt', 'r') as file:
          content = file.read()

      # content = content.replace('load "', 'load "../PDBs/')
      content = content.replace('.pdb', protein + '.pdb')

      if TP[-2] != ',':
          content = content.replace('select TP', TP[:-4])
      if FP[-2] != ',':
          content = content.replace('select FP', FP[:-4])
      if FN[-2] != ',':
          content = content.replace('select FN', FN[:-4])
      
      if is_extra_proteins:
         with open(project_path + '/visualization_scripts/extra_test_proteins_pdb_' + mlp_model + '/' + protein + '.pml', 'w') as file:
            file.write(content)
      else:
        with open(project_path + '/visualization_scripts/test_proteins_pdb_' + mlp_model + '/' + protein + '.pml', 'w') as file:
            file.write(content)
      
def train_classifier(dataset, save_model=False):
  all_proteins = dataset.pdb.unique()

  dreamm_proteins = ['1JOC', '1VFY', '1H0A', '1IAZ', '2AYL', '1GYG', '2P0M', '2FNQ', '3IIQ', '1PFO',
 '1DVP', '1NL1', '5F0P', '1C1Z', '1OIZ', '4X08', '2MH1', '1LA4', '1GWY', '2KS4',
 '1ES6', '3FSN', '1S6X', '1FFJ', '2DDR', '3RZN']

  proteins = [item for item in all_proteins if item not in dreamm_proteins]

  # split number of proteins
  train_proteins, test_proteins = train_test_split(proteins, test_size=0.2, random_state=42)
  test_proteins, val_proteins = train_test_split(test_proteins, test_size=0.5, random_state=42)

  df_train = dataset[dataset.pdb.isin(train_proteins)]
  df_test = dataset[dataset.pdb.isin(test_proteins)]
  df_val = dataset[dataset.pdb.isin(val_proteins)]
  df_extra = dataset[dataset.pdb.isin(dreamm_proteins)]

  X_train, y_train = df_train.drop(['pdb', 'residue_1l', 'residue_number', 'chain_id', 'label'], axis=1, inplace=False).to_numpy(), df_train['label'].values
  X_test, y_test = df_test.drop(['pdb', 'residue_1l', 'residue_number', 'chain_id', 'label'], axis=1, inplace=False).to_numpy(), df_test['label'].values
  X_val, y_val = df_val.drop(['pdb', 'residue_1l', 'residue_number', 'chain_id', 'label'], axis=1, inplace=False).to_numpy(), df_val['label'].values
  X_extra_val, y_extra_val = df_extra.drop(['pdb', 'residue_1l', 'residue_number', 'chain_id', 'label'], axis=1, inplace=False).to_numpy(), df_extra['label'].values

  ratio = df_train.label.value_counts()[0]/df_train.label.value_counts()[1]
  
  # Define the objective function to be maximized.
  def objective(trial):
      
      params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 31, step=2),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", ratio-5, ratio+5),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
        "num_iterations": trial.suggest_int("num_iterations", 100, 2000, step=100),
        'verbosity': -1,
        'device_type': 'cpu',
        'random_state': 42,
        'n_jobs': -1
    }
      model = lgb.train(params, lgb.Dataset(X_train, y_train))

      preds = model.predict(X_val)
      pred_labels = np.rint(preds)

      f1 =  f1_score(y_val, pred_labels)
      mcc =  matthews_corrcoef(y_val, pred_labels)
      conf_matrix = confusion_matrix(y_val, pred_labels)

      # trials[trial.number] = [f1, mcc, conf_matrix]    
      return f1

  # Create a study object and optimize the objective function.
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=200)

  print("Number of finished trials: ", len(study.trials))
  print("Best trial:")
  trial = study.best_trial

  print("  Value: {}".format(trial.value))
  print("  Params: ")
  for key, value in trial.params.items():
      print("    {}: {}".format(key, value))

  clf = lgb.LGBMClassifier(device_type= 'cpu',
    verbosity= -1,
    random_state= 42,
    n_jobs= -1, 
    **study.best_trial.params
    )

  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)

  predicted_df = df_test[['pdb', 'residue_number', 'chain_id', 'label']]
  predicted_df['predicted'] = y_pred

  print()
  print("----------- Model ---------")
  print(mlp_model, dataset_type)
  print()
  print('F1 score: %.3f ' % f1_score(y_test, y_pred))
  print('MCC: %.3f ' % matthews_corrcoef(y_test, y_pred))
  cm = confusion_matrix(y_test, y_pred)
  cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
  print(cm)

  predicted = clf.predict(X_extra_val)

  print('F1 score extra vall: %.3f ' % f1_score(y_extra_val, predicted))

  df_predicted = df_extra[['pdb', 'residue_number', 'chain_id', 'label']]
  df_predicted['predicted'] = predicted

  if save_model:
    file = f'trained_model_{mlp_model}.pkl'
    joblib.dump(clf, open(file, 'wb'))

  return clf

# ---------------------------------------------------
#                      Main
# ---------------------------------------------------
if __name__ == "__main__":
  if mode == 'create_pdb_labeling':
      print("--------------")
      print("Starting generation of embeddings for model", mlp_model )
      print("--------------")
      peprd2ds = read_peprd2ds(project_path + '/Datasets/PePr2DS.csv')
      dreamm = read_dreamm(project_path + '/Datasets/dreamm.json')
      dataset = concatenate_datasets(peprd2ds, dreamm)
      dataset = add_false_labels(dataset)

      # add a column with the 1-letter amino acid name based on the 3-letter name from residue_name column
      dataset['residue_1l'] = dataset['residue_name'].map(amino_acids_3l_key)

      save_dataset(dataset, project_path + f'/Datasets/PDB_dataset/pdb_dataset_{dataset_type}.csv')
      print('Dataset created and saved')

  elif mode == 'create_fasta_file':
      print("--------------")
      print("Starting generation of fasta file" )
      print("--------------")
      df = load_dataset(project_path + f'/Datasets/PDB_dataset/pdb_dataset_{dataset_type}.csv')

      create_fasta_file(df, f'/Datasets/PDB_dataset/{dataset_type}_pdb.fasta')

      print('Fasta files created and saved')

  elif mode == 'filter_proteins':
      print("--------------")
      print("Starting filtering of proteins" )
      print("--------------")
      
      filtered_pdb = keep_representative_pdb(project_path + f'/Datasets/PDB_dataset/cdhit_results_{dataset_type}.txt')

      # read the dataset
      df = load_dataset(project_path + f'/Datasets/PDB_dataset/pdb_dataset_{dataset_type}.csv')

      # keep only the pdb that are in the filtered_pdb list
      filtered_df = df[df['pdb'].isin(filtered_pdb.keys())]

      # keep only the specific chain_id for each pdb
      filtered_df = filtered_df[filtered_df['chain_id'] == filtered_df['pdb'].map(filtered_pdb)]
      
      # save the filtered dataset
      save_dataset(filtered_df, project_path + f'/Datasets/PDB_dataset/pdb_dataset_{dataset_type}_filtered.csv')

      print('Proteins filtered and saved')
    
  elif mode == 'generate_embeddings':
      print("--------------")
      print("Starting generation of embeddings for model", mlp_model )
      print("--------------")
      df = load_dataset(project_path + f'/Datasets/PDB_dataset/pdb_dataset_{dataset_type}_filtered.csv')
      model, tokenizer, num_hidden_size, num_hidden_layers, num_attention_heads = load_model(mlp_model)
      embeddings = generate_embeddings(df, model, tokenizer)
      df = create_embeddings_dataset(df, embeddings, num_hidden_size)
      save_dataset(df, project_path + f'/Datasets/PDB_dataset/pdb_dataset_{dataset_type}_{mlp_model}.csv')

  elif mode == 'train_algorithm':
    print("--------------")
    print("Starting training of algorithms for model", mlp_model )
    print("--------------")

    df = load_dataset(project_path + f'/Datasets/PDB_dataset/pdb_dataset_{dataset_type}_{mlp_model}.csv')

    # one-hot encode of residues
    df_res = pd.get_dummies(df['residue_1l'])
    df = df.merge(df_res, left_index=True, right_index=True, how='inner')

    clf = train_classifier(df, save_model=True)

  elif mode == "predict_new_proteins":
    clf = load_ml_model(f'/home/dimitra/master_thesis/models/lgbm_model_pdb_{mlp_model}.pkl')
    
    sequence_df = get_proteins_sequences(new_proteins)
    sequence_df['residue_1l'] = sequence_df['residue_name'].map(amino_acids_3l_key)

    model, tokenizer, num_hidden_size, num_hidden_layers, num_attention_heads = load_model(mlp_model)
    embeddings = generate_embeddings(sequence_df, model, tokenizer)
    df = create_embeddings_dataset(sequence_df, embeddings, num_hidden_size)
    
    df_res = pd.get_dummies(df['residue_1l'])
    df = df.merge(df_res, left_index=True, right_index=True, how='inner')

    predicted = clf.predict(df.drop(['pdb', 'residue_1l', 'residue_number', 'chain_id', 'label'], axis=1, inplace=False).to_numpy())

    df_predicted = df[['pdb', 'residue_number', 'chain_id']]
    df['predicted'] = predicted

    generate_pymols(df, is_extra_proteins=True)
