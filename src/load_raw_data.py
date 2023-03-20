import pyBigWig
from Bio import SeqIO
import numpy as np
import random
import pandas as pd
import math
import json
from train_val_partition import Train_Chromes, Valid_Chromes, Test_Chromes, strands
epi_dct_pvalue = {"GM12878":{"H3K27me3":"ENCFF211VQW","H3K36me3":"ENCFF397UEP","H3K4me3":"ENCFF480KNX","H3K4me1":"ENCFF836XOQ","H3K9me3":"ENCFF952PCS","H3K9ac":"ENCFF688HLG","H3K27ac":"ENCFF798KYP","H3K4me2":"ENCFF213GVI","H3K79me2":"ENCFF667UBI","H4K20me1":"ENCFF073DJT","H2A.Z":"ENCFF992GSC","DNase":"ENCFF960FMM","ATAC-seq":"ENCFF667MDI","CTCF":"ENCFF637RGD","POLR2A":"ENCFF942TZX"},
"HepG2":{"H3K27me3":"ENCFF942QHN","H3K36me3":"ENCFF094ZKB","H3K4me3":"ENCFF645ZUY","H3K4me1":"ENCFF554XSR","H3K9me3":"ENCFF125NHB"}}

# "H3K27ac","H3K9ac","H3K79me2","H3K4me2","H4K20me1","H2A.Z"
# epi_dct_pvalue_other = {"GM12878":{"H3K9ac":"ENCFF688HLG","H3K27ac":"ENCFF798KYP","H3K4me2":"ENCFF213GVI","H3K79me2":"ENCFF667UBI","H4K20me1":"ENCFF073DJT","H2A.Z":"ENCFF992GSC"}}


SPLICEBERT_PATH = "/rhome/ghao004/bigdata/SpliceBERT/models/SpliceBERT-human.510nt"

def _load_histone_modification(cell_name, file_dct):
    # print(file_dct)
    
    histone_modification_dct = {}
    for histone in file_dct[cell_name]:
        url = "/rhome/ghao004/bigdata/lstm_splicing/{cell_name}/{name_prefix}.bigWig".format(cell_name=cell_name,name_prefix=file_dct[cell_name][histone])
        print("load data "+url)
        histone_modification_dct[histone] = pyBigWig.open(url)
    return histone_modification_dct
    
    
def load_histone_modification(cell_name):

    bws = _load_histone_modification(cell_name,epi_dct_pvalue)
    print("finish_loading epigenomic data")
    
    return bws

def load_genome():
    #load everything
    fasta_sequences = SeqIO.parse(open("/rhome/ghao004/bigdata/lstm_splicing/genome/GRCh38.primary_assembly.genome.fa"),'fasta')
    genome = {}
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        genome[name] = sequence
    print("finish loading genome data")
    return genome


    