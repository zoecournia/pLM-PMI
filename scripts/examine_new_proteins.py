import warnings
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, EsmTokenizer, EsmModel
import torch
from Bio import SeqIO, pairwise2
from Bio.PDB.PDBParser import PDBParser
from Bio.Align import substitution_matrices
import json
import keras
import wget
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings('ignore')

# CONSTANT VARIABLES
amino_acids_3l_key = {"ARG": "R", "HIS": "H", "LYS": "K", "ASP": "D", "GLU": "E", "SER": "S", "THR": "T", "ASN": "N", "GLN": "Q", "CYS": "C",
                      "SEC": "U", "GLY": "G", "PRO": "P", "ALA": "A", "VAL": "V", "ILE": "I", "LEU": "L", "MET": "M", "MSE": "M", "PHE": "F", "TYR": "Y", "TRP": "W"}
amino_acids_1l_key = {'R': 'ARG', 'H': 'HIS', 'K': 'LYS', 'D': 'ASP', 'E': 'GLU', 'S': 'SER', 'T': 'THR', 'N': 'ASN', 'Q': 'GLN', 'C': 'CYS',
                      'U': 'SEC', 'G': 'GLY', 'P': 'PRO', 'A': 'ALA', 'V': 'VAL', 'I': 'ILE', 'L': 'LEU', 'M': 'MET', 'F': 'PHE', 'Y': 'TYR', 'W': 'TRP'}

# Fixed variables
has_gpu = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
num_hidden_size = 0 # 1024  for protTrans or 1280 - will be set automatically

# Variables to define
model_name = 'protTrans' # 'protTrans' or 'esm'

# read proteins that we want to examine
f = open('../extra_proteins/proteins.json')
new_proteins = json.load(f)
f.close()

# Download all proteins - Uniprot + PDB
def fetch_proteins():
    url_for_pdbs = "https://files.rcsb.org/download/"
    url_for_uniprot = "https://rest.uniprot.org/uniprotkb/"

    dest_folder = "./extra_proteins/proteins"

    for uniprot_id in new_proteins.keys():
        # check if uniprot file exists
        if not os.path.exists(dest_folder + "/" + uniprot_id + ".fasta"):
            wget.download(url=url_for_uniprot + uniprot_id +
                          ".fasta", out=dest_folder)

        # check if pdb file exists
        if not os.path.exists(dest_folder + "/" + list(new_proteins[uniprot_id].keys())[0].lower() + ".pdb"):
            wget.download(
                url=url_for_pdbs + list(new_proteins[uniprot_id].keys())[0] + ".pdb", out=dest_folder)


# Sequence alignment function
def seq_alignment(seq1, seq2):
    matrix = substitution_matrices.load("BLOSUM62")

    gap_open = -10
    gap_extend = -1

    alignments = pairwise2.align.localds(
        seq1, seq2, matrix, gap_open, gap_extend)

    return alignments


def align_proteins():
    uniprot_proteins = {}
    for uniprot_id in new_proteins.keys():
        uniprot_proteins[uniprot_id] = []
        # add uniprot sequence
        fasta_file_path = './extra_proteins/proteins/' + uniprot_id + '.fasta'

        for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
            uniprot_proteins[uniprot_id].append(
                {'sequence': str(seq_record.seq)})

        already_existing_aligned_seq = []

        # add pdb sequence
        pdb_id = list(new_proteins[uniprot_id].keys())[0]
        chains = new_proteins[uniprot_id][list(
            new_proteins[uniprot_id].keys())[0]]

        structure = PDBParser().get_structure(
            pdb_id.lower(), './extra_proteins/proteins/' + pdb_id.lower() + '.pdb')
        model = structure[0]
        for ch in model:
            pdb_res_num = []
            seq_pdbParser = ''

            # check if we PePr dataset has this chain. If yes, save the sequence and the indexing
            if ch.id in chains:
                chain = model[ch.id]

                for i in chain.get_residues():
                    # print(i.resname)
                    if i.resname in list(amino_acids_3l_key.keys()):
                        seq_pdbParser += amino_acids_3l_key[i.resname]
                        pdb_res_num.append(
                            i.resname + '' + str(i.get_full_id()[3][1]))

                # align this chain sequence with the uniprot
                pdb_aligned = seq_alignment(
                    uniprot_proteins[uniprot_id][0]['sequence'], seq_pdbParser)

                if pdb_aligned[0][1] not in already_existing_aligned_seq:
                    uniprot_proteins[uniprot_id].append(
                        {pdb_id + '_' + ch.id: []})
                    uniprot_proteins[uniprot_id][len(
                        uniprot_proteins[uniprot_id])-1][pdb_id + '_' + ch.id].append({'seq': pdb_aligned[0][1]})
                    uniprot_proteins[uniprot_id][len(
                        uniprot_proteins[uniprot_id])-1][pdb_id + '_' + ch.id].append({'uni_seq': pdb_aligned[0][0]})
                    uniprot_proteins[uniprot_id][len(
                        uniprot_proteins[uniprot_id])-1][pdb_id + '_' + ch.id].append({'res_num': pdb_res_num})
                    already_existing_aligned_seq.append(pdb_aligned[0][1])

    return uniprot_proteins


def get_proteins_with_sequence(proteins_dict: dict, return_df):
    temp_dict = {}

    for protein in proteins_dict:
        temp_dict[protein] = proteins_dict[protein][0]['sequence']

    if return_df:
        df_temp = pd.DataFrame.from_dict(temp_dict, orient='index', columns=[
            'sequence']).reset_index()
        return df_temp.rename(columns={'index': 'uniprot_id'})
    else:
        return temp_dict


def generate_embeddings(proteins_df, mlp_model):
    results = dict()

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

    proteins_with_seq = proteins_df[['uniprot_id', 'sequence']].groupby('uniprot_id').apply(lambda x: list(np.unique(x['sequence']))[
        0].replace('U', 'X').replace('Z', 'X').replace('O', 'X')).to_dict()

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(proteins_with_seq.items(), key=lambda kv: len(
        proteins_with_seq[kv[0]]), reverse=True)
    
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)

        # get also the chain_id of the specific sequence, by matching the pdb_id and the sequence
        chain_id = proteins_df[(proteins_df['pdb'] == pdb_id) & (proteins_df['sequence'] == seq)].chain_id.unique()[0]

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


def create_df_with_embeddings(proteins_df, embeddings, n_embeddings):
    all_columns = ['uniprot_id', 'residue_1l']
    feat_columns = []

    for i in range(0, n_embeddings):
        all_columns.append('Feature_' + str(i + 1))
        feat_columns.append('Feature_' + str(i + 1))

    df_embeddings = pd.DataFrame(columns=all_columns)

    for protein in embeddings.keys():
        if n_embeddings == 1024:
            temp_df = pd.DataFrame(np.array(embeddings[protein]).reshape(np.array(
                embeddings[protein]).shape[0], np.array(embeddings[protein]).shape[2]), columns=feat_columns)
        else:
            temp_df = pd.DataFrame(
                np.array(embeddings[protein]), columns=feat_columns)
        temp_df.insert(0, 'uniprot_id', protein)
        res_list = [i for i in proteins_df[proteins_df.uniprot_id ==
                                           protein].sequence.unique()[0]]

        temp_df.insert(1, 'residue_1l', res_list)

        df_embeddings = pd.concat(
            [df_embeddings, temp_df], ignore_index=True, sort=False)

    return df_embeddings


def predict_ibs(df_proteins, model_name):
    model = keras.models.load_model('./models/best_model_' + model_name)

    df_res = pd.get_dummies(df_proteins['residue_1l'])
    df_proteins = df_proteins.merge(
        df_res, left_index=True, right_index=True, how='inner')

    ypred = model.predict(df_proteins.drop(
        ['uniprot_id', 'residue_1l'], axis=1, inplace=False))

    return [1 if i > 0.5 else 0 for i in ypred]


def generate_pymol_scripts(df_proteins, aligned_proteins):
    for prot in df_proteins.uniprot_id.unique():
        pdb_seq = ''
        current_len = 100000

        ibs_df = df_proteins[df_proteins.uniprot_id == prot][['predicted']]

        for pdb in aligned_proteins[prot][1:]:
            if pdb[list(pdb.keys())[0]][0]['seq'].count('-') < current_len:
                pdb_seq = pdb
                current_len = pdb[list(pdb.keys())[0]][0]['seq'].count('-')

        TP = 'select TP, '
        residues = pdb_seq[list(pdb_seq.keys())[0]][2]['res_num']
        residues_index = 0

        if '-' in pdb_seq[list(pdb_seq.keys())[0]][0]['seq'] and '-' not in pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']:
            for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][0]['seq']):
                if char != '-':
                    if ibs_df.iloc[ind].predicted == 1:
                        TP += 'chain ' + \
                            list(pdb_seq.keys())[
                                0][5:] + ' and resid ' + str(residues[residues_index][3:]) + ' or '
                    residues_index += 1
                elif ibs_df.iloc[ind].predicted == 1:
                    print(
                        "protein that has penetrating amino acids in other position of sequence " + prot)
            else:
                indx = 0
                pdb_seq_aligned = pdb_seq[list(pdb_seq.keys())[0]][0]['seq']
                for ind, char in enumerate(pdb_seq[list(pdb_seq.keys())[0]][1]['uni_seq']):
                    if char != '-' and pdb_seq_aligned[ind] != '-':
                        if ibs_df.iloc[indx].predicted == 1:
                            TP += 'chain ' + \
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
            print(list(pdb_seq.keys())[0][:-2].lower() + " " + prot)

        # Write new content in write mode 'w'
        with open('./extra_proteins/' + model_name + "/" + list(pdb_seq.keys())[0][:-2] + '.pml', 'w') as file:
            file.write(content)


fetch_proteins()
proteins_alignment = align_proteins()
df_protein = get_proteins_with_sequence(proteins_alignment, return_df=True)

embeddings = generate_embeddings(df_protein, model_name)

df_proteins_with_embeddings = create_df_with_embeddings(
    df_protein, embeddings, num_hidden_size)

predicted_values = predict_ibs(
    df_proteins_with_embeddings, model_name)

df_proteins_with_embeddings['predicted'] = predicted_values

generate_pymol_scripts(df_proteins_with_embeddings, proteins_alignment)
