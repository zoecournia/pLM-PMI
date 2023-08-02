#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:22:54 2021

@author: alexis
"""

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import itertools
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

df2 = pd.read_csv("/Users/dparanou/Master/master_thesis/all_proteins.csv", low_memory=False)
df2 = df2['PDB']

df = pd.DataFrame(index=df2,columns=df2)

matrix = matlist.blosum62

gap_open = -10
gap_extend = -1

with open('/Users/dparanou/Master/master_thesis/all_proteins.fasta', 'r') as f:
    seqs = []
    for line in f:
        if not (line.startswith('>') or line in ('\n', '\r\n')):
            seqs.append(line.strip())

combos = itertools.combinations(seqs, 2)
results = []

print("Before for with combods")

for k,v in tqdm(combos):
    alignments = pairwise2.align.localds(k, v, matrix, gap_open, gap_extend)
    
    align1 = [char for char in alignments[0][0]]
    align2 = [char for char in alignments[0][1]]
    
    matches1 = list(re.finditer('-+', alignments[0][0]))
    matches2 = list(re.finditer('-+', alignments[0][1]))
    if align1[0] == '-':
        start1 = matches1[0].end()
    else:
        start1 = 0
    if align1[-1] == '-':
        end1 = matches1[-1].start()
    else:
        end1 = len(align1)
    if align2[0] == '-':
        start2 = matches2[0].end()
    else:
        start2 = 0
    if align2[-1] == '-':
        end2 = matches2[-1].start()
    else:
        end2 = len(align2)
    
    tga = end1 - start1
    tgb = end2 - start2
    
    i = 0
#    j = 0
    for a in range(0,len(align1)):
        if align1[a] == align2[a]:
            i += 1
# =============================================================================
#         if align1[a] != '-' and align2[a] != '-':
#             j += 1
#     results.append(100*i/j)
# =============================================================================
    #results.append(100*i/len(min(k, v, key=len)))
    results.append(100*i/min(tga, tgb))
                   
kkk = list(itertools.combinations(enumerate(df2), 2))

for l, (i,j) in enumerate(kkk):
    df[i[1]][j[1]] = results[l]
    df[j[1]][i[1]] = results[l]

#df.values[[np.arange(df.shape[0])]*2] = 0
df = df.apply(pd.to_numeric)
df = df.round(2)

df.to_csv('pair_results.csv')
#for i, row in df.iterrows():
#    print(i, row.max(), row.idxmax(axis=1))

