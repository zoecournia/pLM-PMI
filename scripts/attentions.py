import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, EsmTokenizer, EsmModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

import torch
import tensorflow as tf

has_gpu = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" 

project_path = '/home/dimitra/master_thesis/'
# ---------------------------------------------------
#                      Variables
# ---------------------------------------------------
mlp_model = 'esm' # esm, protTrans
dataset_type = 'mean' # mean, apc, filtered
mode = 'train_classifier' # plot_attention_maps, train_classifier, generate_attention_maps
set_to_examine = 'val_dreamm' # it works only for this set

# ---------------------------------------------------
#                      Functions 
# ---------------------------------------------------

def load_dataset(path):
  # Load dataset
  df = pd.read_csv(path, index_col=0)

  # remove sequences with length > 1024 and sequences with no IBS
  valid_seq = []
  for seq in df.sequence.unique():
      if len(seq) < 2048: # and sum(df[df.sequence == seq].is_IBS.values) != 0:
        valid_seq.append(df[df.sequence == seq].uniprot_id.unique()[0])

  df = df[df.uniprot_id.isin(valid_seq)]
  return df

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

  # Get the number of hidden layers and attention heads
  num_hidden_layers = model.config.num_hidden_layers
  num_attention_heads = model.config.num_attention_heads

  print('Model successfully loaded and running in',device)

  return model, tokenizer, num_hidden_layers, num_attention_heads

'''
Get the number of hidden layers and attention heads
Returns the column names that is used -> [0_0, 0_1, ..., N_K-1, N_K]
Ex. [layer0_head0, layer0_head1, ..., leayerN_headK-1, layerN_headK]
'''
def create_col_names(num_hidden_layers, num_attention_heads):
  col_names = []
  # make column names
  for i in range(num_hidden_layers):
    for j in range(num_attention_heads):
      col_names.append(f"{i}_{j}")
  return col_names

def calculate_apc(matrix):
    F_i = np.sum(matrix, axis=1)
    F_j = np.sum(matrix, axis=0)

    F = np.sum(matrix)

    F_ij_apc = matrix - np.outer(F_i,F_j) / F

    return F_ij_apc

def create_att_maps_df(attention_weights, attentions_df, num_hidden_layers, num_attention_heads):
  for layer in range(num_hidden_layers):
    for head in range(num_attention_heads):
      # create df with APC -> mean of rows
      if dataset_type == 'apc':
        apc_att_map = calculate_apc(attention_weights[layer][head,:,:])
        attentions_df[str(layer) + "_" + str(head)] = np.mean(apc_att_map, axis=0)
      
      # create df with only mean col
      elif dataset_type == 'mean':
        attentions_df[str(layer) + "_" + str(head)] = np.mean(attention_weights[layer][head,:,:], axis=0)
      
      # create df with filter weightes > 0.3 & mean col
      else:
        # Calculate the number of cells in each column that are greater than 0.3
        num_cells_greater_than_0_3 = np.sum(attention_weights[layer][head,:,:] > 0.3, axis=0)

        # Calculate the total number of values greater than 0.3 in the whole matrix
        total_values_greater_than_0_3 = np.sum(attention_weights[layer][head,:,:] > 0.3)

        # Ensure the denominator is not zero
        if total_values_greater_than_0_3 != 0:
          # Calculate the division
          attentions_df[str(layer) + "_" + str(head)] = num_cells_greater_than_0_3 / total_values_greater_than_0_3
        else:
          # Handle the case when there are no values greater than 0.3
          attentions_df[str(layer) + "_" + str(head)] = np.zeros(attention_weights[layer][head,:,:].shape[1])  # Or handle as desired

  return attentions_df

def train_classifier(dataset): # , num_hidden_layers, num_attention_heads):
    # Identify unique proteins in the dataset
    proteins = dataset.groupby('uniprot').filter(lambda x: x['label'].sum() != 0)['uniprot'].unique()

    # Filter the dataset to include only sequences with IBS
    dataset = dataset[dataset['uniprot'].isin(proteins)]

    proteins = dataset.uniprot.unique()

    # split number of proteins
    train_proteins, test_proteins = train_test_split(proteins, test_size=0.2, random_state=42)

    df_train = dataset[dataset.uniprot.isin(train_proteins)]
    df_test = dataset[dataset.uniprot.isin(test_proteins)]

    X_train, y_train = df_train.drop(['uniprot', 'residue', 'label'], axis=1, inplace=False).to_numpy(), df_train['label'].values
    X_test, y_test = df_test.drop(['uniprot', 'residue', 'label'], axis=1, inplace=False).to_numpy(), df_test['label'].values

    ratio = df_train.label.value_counts()[0]/df_train.label.value_counts()[1]
   
    clf = lgb.LGBMClassifier(objective='binary', 
                            #  is_unbalance=True, 
                            #  class_weight='balanced', 
                             scale_pos_weight=np.sqrt(ratio),
                             device_type="gpu")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print()
    print("----------- Model ---------")
    print(mlp_model , dataset_type)
    print()
    print('F1 score: %.3f ' % f1_score(y_test, y_pred))
    print('MCC: %.3f ' % matthews_corrcoef(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])
    print(cm)

    # heat_map = np.zeros((num_hidden_layers+1, num_attention_heads+1)) # ESM

    # layer = 0
    # head = 0
    # for coe in clf_sgd.coef_[0]:
    #     if head == num_attention_heads: # ESM
    #         heat_map[layer][head] = abs(coe)
    #         head = 0
    #         layer +=1
    #     else:
    #         heat_map[layer][head] = abs(coe)
    #         head += 1

    # ax = sns.heatmap(heat_map, cmap="Blues", linewidth=.5, fmt=".2f%", cbar=False)
    # ax.invert_yaxis()
    # cbar = ax.figure.colorbar(ax.collections[0])
    # cbar.set_ticks([i for i in range(0, int(round(heat_map.max(), -1) + 10), 5)])
    # cbar.set_ticklabels([str(i) + "" for i in range(0, int(round(heat_map.max(), -1) + 10), 5)])

    # plt.xlabel('Attention head')
    # plt.ylabel('Hidden layer')
    # plt.title('Coefficients of SGD (stochastic gradient descent)')
    # plt.savefig('my_plot.png', dpi=300)

    return clf

'''
Get the uniprot ids of train / test / val / extra_test sets
'''
def get_sets_of_protein():
    f = open('./Datasets/My_Dataset/split_proteins.json')
    dict_proteins_split = json.load(f)
    f.close()

    return dict_proteins_split

'''
Get file with the uniprot sequence aligned with the corresponding pdbs sequences
'''
def get_aligned_proteins(which_set, all_splits):
    f = open('./Datasets/My_Dataset/aligned_results.json')
    dict_proteins_aligned = json.load(f)
    f.close()

    prot_dict = {}
    for prot in all_splits[which_set]:
        prot_dict[prot] = dict_proteins_aligned[prot]

    return prot_dict

'''
Get dataframe with only proteins for a specific set (train / test / etc.)
'''
def get_specific_proteins_df(which_set, all_splits, df):
    return df[df.uniprot.isin(all_splits[which_set])]

'''
Generate pymol files for visualisation for a specific set
'''
def generate_pymol_script(attentions_df, mlp_model, clf):
  dataset_split = get_sets_of_protein()
  aligned_proteins = get_aligned_proteins(set_to_examine, dataset_split)
  df_splitted = get_specific_proteins_df(set_to_examine, dataset_split, attentions_df)

  # create folder if not exists
  if not os.path.exists(project_path + 'visualization_scripts/attentions_' + set_to_examine + '_proteins_' + mlp_model):
        os.mkdir(project_path + 'visualization_scripts/attentions_' + set_to_examine + '_proteins_' + mlp_model)
  
  df_X = df_splitted.drop(['uniprot', 'residue', 'label'], axis=1, inplace=False)
  y_pred = clf.predict(df_X)

  for prot in df_splitted.uniprot.unique():
      ibs_df = df_splitted[df_splitted.uniprot == prot][['label']]

      if (y_pred == 0).all():
          print("All predictions are 0 " + prot)

      # set initial values for the pdb sequence that is closer to the uniprot sequence
      pdb_seq = ''
      current_len = 100000

      # find the sequence which is closer to uniprot seq
      for pdb in aligned_proteins[prot][1:]:
          # for pdb in test_prot_dict[prot]:
          if pdb[list(pdb.keys())[0]][0]['seq'].count('-') < current_len:
              pdb_seq = pdb
              current_len = pdb[list(pdb.keys())[0]][0]['seq'].count('-')

      # get the specific sequence
      residues = pdb_seq[list(pdb_seq.keys())[0]][2]['res_num']
      residues_index = 0

      TP = 'select TP, '

      if '-' in pdb_seq[list(pdb_seq.keys())[0]][0]['seq'] and '-' not in pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']:
          for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][0]['seq']):
              if char != '-':
                  if ibs_df.iloc[ind].label == 0 and y_pred[ind] == 1:
                      TP += 'chain ' + \
                          list(pdb_seq.keys())[
                              0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                  residues_index += 1
              elif y_pred[ind] == 1:
                  print(
                      "protein that has penetrating amino acids in other position of sequence " + prot)
      
      # create pymol script
      with open(project_path + 'scripts/pymol_template.txt', 'r') as file:
          content = file.read()
          
      content = content.replace('load "', 'load "../PDBs/')
      content = content.replace('.pdb', list(pdb_seq.keys())[
          0][:-2].lower() + '.pdb')

      # print(temp_str)
      print(list(pdb_seq.keys())[0][:-2])
      # check that they have values
      if TP[-2] != ',':
          content = content.replace('select TP', TP[:-4])
      else:
          print("No TP values for " + list(pdb_seq.keys())
                [0][:-2].lower() + " " + prot)

      # Write new content in write mode 'w'
      with open(project_path + 'visualization_scripts/attentions_' + set_to_examine + '_proteins_' + mlp_model + '/' + list(pdb_seq.keys())[0][:-2] + '.pml', 'w') as file:
          file.write(content)

      # move pdb files
      # shutil.copyfile(project_path + " /Datasets/My_Dataset/Proteins_PDB/" + list(pdb_seq.keys())[0][:-2].lower(
      # ) + '.pdb', './visualization_scripts/' + set_to_examine + '_proteins_' + model_name + '/' + list(pdb_seq.keys())[0][:-2].lower() + '.pdb')
               
'''
Convert appropriatly the attention maps from tuples to arrays
'''
def format_attention_maps(attentions):
    attention_weights = []
    for layer in attentions:
        layer_attention_weights = []
        for head in layer:
            layer_attention_weights.append(head.detach().cpu().numpy())
        attention_weights.append(layer_attention_weights)
    attention_weights = np.squeeze(attention_weights, axis=1)
    
    return attention_weights

'''
Calculate the proportion of attention maps based on mathematical formula from paper "Bertology..."
'''
def calculate_p(label, attentionmap, th):
    l, h, i, j = attentionmap.shape
    numerator = np.zeros((l, h))
    denominator = np.zeros((l, h))

    # Apply threshold to attentionmap
    attentionmap_thresholded = np.where(attentionmap > th, 1, 0)
    
    attentionmap_mask = attentionmap > th

    for ll in range(l):
        for hh in range(h):
            numerator[ll][hh] = np.sum(label * attentionmap_thresholded[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
            denominator[ll][hh] = np.sum(attentionmap_thresholded[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
    return numerator, denominator

# -------------------- Generate attention maps for creating plots of weights --------------------
def plot_attention_maps(df, num_hidden_layers, num_attention_heads, model, tokenizer, mlp_model):
  # keep proteins with their sequences
  proteins_with_seq = df[['uniprot_id', 'sequence']].groupby('uniprot_id').apply(lambda x: list(np.unique(x['sequence']))[0].replace('U','X').replace('Z','X').replace('O','X')).to_dict()

  thresholds = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
  fig, axs = plt.subplots(2, 3, figsize=(20, 10))
  index = 0

  for threshold in tqdm(thresholds):
    print("Threshold ", threshold)
    grand_numerator = np.zeros((num_hidden_layers, num_attention_heads))
    grand_denominator = np.zeros((num_hidden_layers, num_attention_heads))
    probability = np.zeros((num_hidden_layers, num_attention_heads))

    for seq_idx, (pdb_id, seq) in tqdm(enumerate(proteins_with_seq.items())):
      seq = seq
      seq_len = len(seq)

      if mlp_model == 'protTrans':
        seq = ' '.join(list(seq)) # add space between each AA for protTrans model

      token_encoding = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)

      try:
        with torch.no_grad():
          embedding_repr = model(**token_encoding, output_attentions=True)

      except RuntimeError:
        print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
        continue
      
      attention_weights = format_attention_maps(embedding_repr.attentions)
      label = df[df.uniprot_id == pdb_id]['is_IBS'].to_numpy()
      y = np.tile(label, (len(label), 1))

      # calculate the numerator of the proportion
      numerator, denominator = calculate_p(y, attention_weights, threshold)
      grand_numerator += numerator
      grand_denominator += denominator

    for l in range(num_hidden_layers):
        for h in range(num_attention_heads):
            probability[l][h] = grand_numerator[l][h] / grand_denominator[l][h] if grand_denominator[l][h] != 0 else 0
    probability = probability * 100

    if index < 3:
        row = 0
        col = index
    else:
        row = 1
        col = index - 3
    index += 1

    ax = sns.heatmap(probability, cmap="Blues", linewidth=.5, fmt=".2f%", cbar=False, ax=axs[row, col])
    ax.invert_yaxis()
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks([i for i in range(0, int(round(probability.max(), -1) + 10), 10)])
    cbar.set_ticklabels([str(i) + "%" for i in range(0, int(round(probability.max(), -1) + 10), 10)])
    axs[row, col].set_xlabel('Attention head')
    axs[row, col].set_ylabel('Hidden layer')
    axs[row, col].set_title('Proportion of attention threshold ' + str(threshold))

  plt.savefig('attention_weights_' + mlp_model + '.png', dpi=300)

# -------------------- Generate attention maps for ML training --------------------   
def generate_attention_maps(df, num_hidden_layers, num_attention_heads, model, tokenizer, mlp_model):
  attentions_df = pd.DataFrame()
  col_names = create_col_names(num_hidden_layers, num_attention_heads)

  # keep proteins with their sequences
  proteins_with_seq = df[['uniprot_id', 'sequence']].groupby('uniprot_id').apply(lambda x: list(np.unique(x['sequence']))[0].replace('U','X').replace('Z','X').replace('O','X')).to_dict()
  results = dict()

  # prepare df with attention maps
  for seq_idx, (uniprot_id, seq) in tqdm(enumerate(proteins_with_seq.items())):
      seq = seq
      seq_len = len(seq)

      if mlp_model == 'protTrans':
        seq = ' '.join(list(seq)) # add space between each AA for protTrans model

      token_encoding = tokenizer(seq, return_tensors='pt', add_special_tokens=False).to(device)

      try:
        with torch.no_grad():
          embedding_repr = model(**token_encoding, output_attentions=True)

      except RuntimeError:
        print("RuntimeError during embedding for {} (L={})".format(uniprot_id, seq_len))
        continue
      
      attention_weights = format_attention_maps(embedding_repr.attentions)
      label = df[df.uniprot_id == uniprot_id]['is_IBS'].to_numpy()
      y = np.tile(label, (len(label), 1))

      temp_df = pd.DataFrame(columns=col_names)
      temp_df.insert(0, "residue", df[df.uniprot_id == uniprot_id]['residue_1l'].values)
      temp_df.insert(1, "uniprot", df[df.uniprot_id == uniprot_id]['uniprot_id'].values)
      temp_df.insert(2, "label", df[df.uniprot_id == uniprot_id]['is_IBS'].values)
      temp_df = create_att_maps_df(attention_weights, temp_df, num_hidden_layers, num_attention_heads)  

      attentions_df = pd.concat([attentions_df, temp_df], ignore_index=True)
  
  attentions_df.to_csv("attentions_" + dataset_type + "_" + mlp_model + ".csv")

# -------------------- Train classifier --------------------
def train_classifier_from_attention_maps(file_path, mlp_model):
  # get file with attention maps as features
  attentions_df = pd.read_csv(file_path)
  # train ml classifier
  clf = train_classifier(attentions_df)
  # generate pymol files for proteins
  # generate_pymol_script(attentions_df, mlp_model, clf)


# ---------------------------------------------------
#                      Main
# ---------------------------------------------------
  
if __name__ == "__main__":
# -------------------- Generate attention maps for creating plots of weights --------------------
  if mode == 'plot_attention_maps':
    # -------------------- Load model --------------------
    model, tokenizer, num_hidden_layers, num_attention_heads = load_model(mlp_model)

    # -------------------- Load dataset --------------------
    df = load_dataset(project_path + 'Datasets/My_Dataset/proteins_df_annotated.csv')

    plot_attention_maps(df, num_hidden_layers, num_attention_heads, model, tokenizer, mlp_model)

  # -------------------- Generate attention maps for ML training --------------------
  if mode == 'generate_attention_maps':
    # -------------------- Load model --------------------
    model, tokenizer, num_hidden_layers, num_attention_heads = load_model(mlp_model)

    # -------------------- Load dataset --------------------
    df = load_dataset(project_path + 'Datasets/My_Dataset/proteins_df_annotated.csv')

    generate_attention_maps(df, num_hidden_layers, num_attention_heads, model, tokenizer, mlp_model)

  # -------------------- Train classifier --------------------
  if mode == 'train_classifier':
    # train_classifier_from_attention_maps(num_hidden_layers, num_attention_heads, project_path + "attentions/attentions_" + dataset_type + "_" + mlp_model + ".csv", mlp_model)
    train_classifier_from_attention_maps(project_path + "attentions/attentions_" + dataset_type + "_" + mlp_model + ".csv", mlp_model)