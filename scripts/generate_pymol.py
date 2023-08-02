# import tensorflow as tf
import os
import pandas as pd
import json
import shutil
# from xgboost import XGBClassifier

is_attention_results = True
plm = 'protTrans'  # ['protTrans', 'esm']
# ['val_dreamm', 'val', 'test', 'attentions', 'attentions_val_dreamm']
set_to_examine = 'attentions'
# 0.55 for esm, 0.588 for protTrans, 0.855679 for esm_sl, 0.460432 for protTrans_sl, 0.838641 for protrans_mlp_do
threshold = 0.55
# ['protTrans', 'esm', 'protTrans_sl', 'esm_sl', 'xgboost_esm', 'xgboost_protTrans', 'xgboost_protTrans_2', protTrans_mlp_do]
model_name = 'protTrans'

amino_acids_3l_key = {"ARG": "R", "HIS": "H", "LYS": "K", "ASP": "D", "GLU": "E", "SER": "S", "THR": "T", "ASN": "N", "GLN": "Q", "CYS": "C",
                      "SEC": "U", "GLY": "G", "PRO": "P", "ALA": "A", "VAL": "V", "ILE": "I", "LEU": "L", "MET": "M", "MSE": "M", "PHE": "F", "TYR": "Y", "TRP": "W"}


def load_dataset():
    if is_attention_results:
        df = pd.read_csv(
            './Datasets/My_Dataset/proteins_df_annotated.csv', index_col=0)
        f = open("./attentions/results.json")
        results = json.load(f)
        f.close()

        return df, results
    else:
        df = pd.read_csv('./Datasets/My_Dataset/proteins_embeddings_' +
                         plm + '_annotated.csv', index_col=0)

        # convert residues to one-hot encode
        df_res = pd.get_dummies(df['residue_1l'])

        df = df.merge(df_res, left_index=True, right_index=True, how='inner')
        df.insert(len(df.columns)-1, 'is_IBS', df.pop('is_IBS'))

        return df

# Get the split of the proteins - train, val, test


def get_sets_of_protein():
    f = open('./Datasets/My_Dataset/split_proteins.json')
    dict_proteins_split = json.load(f)
    f.close()

    return dict_proteins_split


# seperate the data to X with only the necessary columns and y - target
def get_dataframe_X_y(df):
    return (df.drop(['uniprot_id', 'residue_1l', 'is_IBS'], axis=1, inplace=False), df['is_IBS'])


# get the specific dataframe for the set - only proteins that are in the set
def get_specific_proteins_df(which_set, all_splits, df):
    return df[df.uniprot_id.isin(all_splits[which_set])]


def get_aligned_proteins(which_set, all_splits):
    f = open('./Datasets/My_Dataset/aligned_results.json')
    dict_proteins_aligned = json.load(f)
    f.close()

    prot_dict = {}
    for prot in all_splits[which_set]:
        prot_dict[prot] = dict_proteins_aligned[prot]

    return prot_dict


def predict_values(X_df):
    if 'xgboost' in model_name:
        model = XGBClassifier()
        model.load_model('./models/' + model_name + '.json')

        ypred = model.predict(X_df)
    else:
        model = tf.keras.models.load_model('./models/best_model_' + model_name)
        ypred_prob = model.predict(X_df)
        ypred = [1 if i > threshold else 0 for i in ypred_prob]
    return ypred


def generate_pymol(which_set, df_X, y_pred, aligned_proteins_dict):
    # create folder if not exists
    if is_attention_results:
        if not os.path.exists('./visualization_scripts/' + set_to_examine + '_proteins_' + model_name):
            os.mkdir('./visualization_scripts/' +
                     set_to_examine + '_proteins_' + model_name)

        for prot in df_X.uniprot_id.unique():
            ibs_df = df_X[df_X.uniprot_id == prot][['is_IBS']]

            if set_to_examine == 'attentions_val_dreamm':
                ibs_df['predicted'] = y_pred[model_name + '_val_dreamm'][prot]
            else:
                ibs_df['predicted'] = y_pred[model_name][prot]

            if (ibs_df['predicted'] == 0).all():
                print("All predictions are 0 " + prot)

            # set initial values for the pdb sequence that is closer to the uniprot sequence
            pdb_seq = ''
            current_len = 100000

            # find the sequence which is closer to uniprot seq
            for pdb in aligned_proteins_dict[prot][1:]:
                # for pdb in test_prot_dict[prot]:
                if pdb[list(pdb.keys())[0]][0]['seq'].count('-') < current_len:
                    pdb_seq = pdb
                    current_len = pdb[list(pdb.keys())[0]][0]['seq'].count('-')

            # get the specific sequence
            residues = pdb_seq[list(pdb_seq.keys())[0]][2]['res_num']
            residues_index = 0

            if which_set == 'attentions_val_dreamm':
                TP = 'select TP, '

                if '-' in pdb_seq[list(pdb_seq.keys())[0]][0]['seq'] and '-' not in pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']:
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][0]['seq']):
                        if char != '-':
                            if ibs_df.iloc[ind].is_IBS == 0 and ibs_df.iloc[ind].predicted == 1:
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                        elif ibs_df.iloc[ind].predicted == 1:
                            print(
                                "protein that has penetrating amino acids in other position of sequence " + prot)
                else:
                    indx = 0
                    pdb_seq_aligned = pdb_seq[list(
                        pdb_seq.keys())[0]][0]['seq']
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']):
                        if char != '-' and pdb_seq_aligned[ind] != '-':
                            if ibs_df.iloc[indx].is_IBS == 0 and ibs_df.iloc[indx].predicted == 1:
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                            indx += 1
                        elif char == '-' and pdb_seq_aligned[ind] != '-':
                            residues_index += 1
            else:
                TP = 'select TP, '
                FP = 'select FP, '
                FN = 'select FN, '

                if '-' in pdb_seq[list(pdb_seq.keys())[0]][0]['seq'] and '-' not in pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']:
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][0]['seq']):
                        if char != '-':
                            if ibs_df.iloc[ind].is_IBS == 1 and ibs_df.iloc[ind].predicted == 1:
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[ind].is_IBS == 0 and ibs_df.iloc[ind].predicted == 1:
                                FP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[ind].is_IBS == 1 and ibs_df.iloc[ind].predicted == 0:
                                FN += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                        elif ibs_df.iloc[ind].predicted == 1:
                            print(
                                "protein that has penetrating amino acids in other position of sequence " + prot)
                else:
                    indx = 0
                    pdb_seq_aligned = pdb_seq[list(
                        pdb_seq.keys())[0]][0]['seq']
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']):
                        if char != '-' and pdb_seq_aligned[ind] != '-':
                            if ibs_df.iloc[indx].is_IBS == 1 and ibs_df.iloc[indx].predicted == 1:
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[indx].is_IBS == 0 and ibs_df.iloc[indx].predicted == 1:
                                FP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[indx].is_IBS == 1 and ibs_df.iloc[indx].predicted == 0:
                                FN += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                            indx += 1
                        elif char == '-' and pdb_seq_aligned[ind] != '-':
                            residues_index += 1

            # create pymol script
            with open('pymol_template.txt', 'r') as file:
                content = file.read()

            content = content.replace('.pdb', list(pdb_seq.keys())[
                0][:-2].lower() + '.pdb')

            # check that they have values
            if TP[-2] != ',':
                content = content.replace('select TP', TP[:-4])
            else:
                print("No TP values for " + list(pdb_seq.keys())
                      [0][:-2].lower() + " " + prot)
            if which_set != 'attentions_val_dreamm':
                if FP[-2] != ',':
                    content = content.replace('select FP', FP[:-4])
                if FN[-2] != ',':
                    content = content.replace('select FN', FN[:-4])

            # Write new content in write mode 'w'
            with open('./visualization_scripts/' + set_to_examine + '_proteins_' + model_name + '/' + list(pdb_seq.keys())[0][:-2] + '.pml', 'w') as file:
                file.write(content)

            # move pdb files
            shutil.copyfile("./Datasets/My_Dataset/Proteins_PDB/" + list(pdb_seq.keys())[0][:-2].lower(
            ) + '.pdb', './visualization_scripts/' + set_to_examine + '_proteins_' + model_name + '/' + list(pdb_seq.keys())[0][:-2].lower() + '.pdb')
    else:
        if not os.path.exists('./visualization_scripts/' + set_to_examine + '_proteins_' + model_name):
            os.mkdir('./visualization_scripts/' +
                     set_to_examine + '_proteins_' + model_name)

        df_X['predicted'] = y_pred

        for prot in df_X.uniprot_id.unique():
            ibs_df = df_X[df_X.uniprot_id == prot][['is_IBS', 'predicted']]

            if (ibs_df['predicted'] == 0).all():
                print("All predictions are 0 " + prot)

            # set initial values for the pdb sequence that is closer to the uniprot sequence
            pdb_seq = ''
            current_len = 100000

            # find the sequence which is closer to uniprot seq
            for pdb in aligned_proteins_dict[prot][1:]:
                # for pdb in test_prot_dict[prot]:
                if pdb[list(pdb.keys())[0]][0]['seq'].count('-') < current_len:
                    pdb_seq = pdb
                    current_len = pdb[list(pdb.keys())[0]][0]['seq'].count('-')

            # get the specific sequence
            residues = pdb_seq[list(pdb_seq.keys())[0]][2]['res_num']
            residues_index = 0

            temp_str = ''
            if which_set == 'val_dreamm':
                TP = 'select TP, '

                if '-' in pdb_seq[list(pdb_seq.keys())[0]][0]['seq'] and '-' not in pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']:
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][0]['seq']):
                        if char != '-':
                            if ibs_df.iloc[ind].is_IBS == 0 and ibs_df.iloc[ind].predicted == 1:
                                temp_str += amino_acids_3l_key[residues[residues_index][:3]] + str(
                                    residues[residues_index][3:]) + ', '
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                        elif ibs_df.iloc[ind].predicted == 1:
                            print(
                                "protein that has penetrating amino acids in other position of sequence " + prot)
                else:
                    indx = 0
                    pdb_seq_aligned = pdb_seq[list(
                        pdb_seq.keys())[0]][0]['seq']
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']):
                        if char != '-' and pdb_seq_aligned[ind] != '-':
                            if ibs_df.iloc[indx].is_IBS == 0 and ibs_df.iloc[indx].predicted == 1:
                                temp_str += amino_acids_3l_key[residues[residues_index][:3]] + str(
                                    residues[residues_index][3:]) + ', '
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                            indx += 1
                        elif char == '-' and pdb_seq_aligned[ind] != '-':
                            residues_index += 1
            else:
                TP = 'select TP, '
                FP = 'select FP, '
                FN = 'select FN, '

                if '-' in pdb_seq[list(pdb_seq.keys())[0]][0]['seq'] and '-' not in pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']:
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][0]['seq']):
                        if char != '-':
                            if ibs_df.iloc[ind].is_IBS == 1 and ibs_df.iloc[ind].predicted == 1:
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[ind].is_IBS == 0 and ibs_df.iloc[ind].predicted == 1:
                                FP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[ind].is_IBS == 1 and ibs_df.iloc[ind].predicted == 0:
                                FN += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                        elif ibs_df.iloc[ind].predicted == 1:
                            print(
                                "protein that has penetrating amino acids in other position of sequence " + prot)
                else:
                    indx = 0
                    pdb_seq_aligned = pdb_seq[list(
                        pdb_seq.keys())[0]][0]['seq']
                    for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']):
                        if char != '-' and pdb_seq_aligned[ind] != '-':
                            if ibs_df.iloc[indx].is_IBS == 1 and ibs_df.iloc[indx].predicted == 1:
                                TP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[indx].is_IBS == 0 and ibs_df.iloc[indx].predicted == 1:
                                FP += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            elif ibs_df.iloc[indx].is_IBS == 1 and ibs_df.iloc[indx].predicted == 0:
                                FN += 'chain ' + \
                                    list(pdb_seq.keys())[
                                        0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                            residues_index += 1
                            indx += 1
                        elif char == '-' and pdb_seq_aligned[ind] != '-':
                            residues_index += 1

            # create pymol script
            with open('pymol_template.txt', 'r') as file:
                content = file.read()

            content = content.replace('.pdb', list(pdb_seq.keys())[
                0][:-2].lower() + '.pdb')

            print(temp_str)
            print(list(pdb_seq.keys())[0][:-2])
            # check that they have values
            if TP[-2] != ',':
                content = content.replace('select TP', TP[:-4])
            else:
                print("No TP values for " + list(pdb_seq.keys())
                      [0][:-2].lower() + " " + prot)
            if which_set != 'val_dreamm':
                if FP[-2] != ',':
                    content = content.replace('select FP', FP[:-4])
                if FN[-2] != ',':
                    content = content.replace('select FN', FN[:-4])

            # Write new content in write mode 'w'
            with open('./visualization_scripts/' + set_to_examine + '_proteins_' + model_name + '/' + list(pdb_seq.keys())[0][:-2] + '.pml', 'w') as file:
                file.write(content)

            # move pdb files
            shutil.copyfile("./Datasets/My_Dataset/Proteins_PDB/" + list(pdb_seq.keys())[0][:-2].lower(
            ) + '.pdb', './visualization_scripts/' + set_to_examine + '_proteins_' + model_name + '/' + list(pdb_seq.keys())[0][:-2].lower() + '.pdb')


if is_attention_results:
    df_full, results = load_dataset()
    dataset_split = get_sets_of_protein()
    aligned_proteins = get_aligned_proteins(set_to_examine, dataset_split)
    df_splitted = get_specific_proteins_df(
        set_to_examine, dataset_split, df_full)

    generate_pymol(set_to_examine, df_splitted, results, aligned_proteins)
else:
    df_full = load_dataset()
    dataset_split = get_sets_of_protein()
    aligned_proteins = get_aligned_proteins(set_to_examine, dataset_split)
    df_splitted = get_specific_proteins_df(
        set_to_examine, dataset_split, df_full)
    X, y = get_dataframe_X_y(df_splitted)
    ypred = predict_values(X)

    generate_pymol(set_to_examine, df_splitted, ypred, aligned_proteins)
