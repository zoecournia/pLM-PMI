import wget
import os
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url_for_pdbs = "https://files.rcsb.org/download/"
url_for_fasta = "https://www.rcsb.org/fasta/entry/"
url_for_uniprot = "https://rest.uniprot.org/uniprotkb/"

# save the path of data folder
dest_folder_uniprot = os.path.realpath("./Datasets/My_Dataset/Proteins_Uniprot_fasta")
dest_folder_pdb = os.path.realpath("./Datasets/My_Dataset/Proteins_PDB")

# load proteins that do not have fasta file -> have been replaced
f = open('./Datasets/My_Dataset/uniprot_proteins.json')
uniprot_proteins = json.load(f)

# for uniprot download
for protein in uniprot_proteins.keys():
  if not os.path.exists(dest_folder_uniprot + "/" + protein + ".fasta"):
    print(protein)
    wget.download(url=url_for_uniprot + protein + ".fasta", out=dest_folder_uniprot)
    print()

f.close()

with open('./Datasets/My_Dataset/pdb_for_download.txt') as file:
  pdb_proteins = [line.rstrip() for line in file]

# for pdb download
for protein in pdb_proteins:
  if not os.path.exists(dest_folder_pdb + "/" + protein.lower() + ".pdb"):
    print(protein)
    wget.download(url=url_for_pdbs + protein + ".pdb", out=dest_folder_pdb)
    print()
