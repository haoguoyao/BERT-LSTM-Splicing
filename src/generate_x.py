import numpy as np
from load_raw_data import load_histone_modification, load_genome,SPLICEBERT_PATH
import numba as nb
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification
from args import args



if args.histone == 'core':
    histone_type_lst = ["H3K27me3","H3K36me3","H3K4me3","H3K4me1","H3K9me3"]
elif args.histone == 'all':
    histone_type_lst = ["H3K27me3","H3K36me3","H3K4me3","H3K4me1","H3K9me3","H3K27ac","H3K9ac","H3K79me2","H3K4me2","H4K20me1","H2A.Z","DNase","ATAC-seq","CTCF","POLR2A"]
elif args.histone == 'none':
    histone_type_lst = []


genome = load_genome()
tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)
class TempData:
    def __init__(self):
        self.cell_type = None
        self.histone_modification = None

    def set(self, cell_type):
        self.cell_type = cell_type
        self.histone_modification = load_histone_modification(cell_type)

tempData = TempData()

# @nb.jit
def one_hot_encode_X(Xd):
    # One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond # to A, C, G, T respectively.
    IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    return IN_MAP[Xd.astype('int8')]



@nb.jit
def reverse_sequence_lst(seq):
    reverse_dct = {"A":"T","T":"A","G":"C","C":"G","N":"N"}
    seq = seq[::-1]   
    sequence_lst = list(seq)
    for i in range(len(sequence_lst)):
        sequence_lst[i] = reverse_dct[sequence_lst[i]]
    return sequence_lst

# @nb.jit
def reverse_histone_mark_lst(histone_mark_lst):
    for i in range(len(histone_mark_lst)):
        histone_mark_lst[i] = histone_mark_lst[i][::-1]
    return histone_mark_lst


@nb.jit
def encode_sequence(sequence_lst):
    int_dct = {"N":0,"A":1,"T":2,"C":3,"G":4}
    for i in range(len(sequence_lst)):
        sequence_lst[i] = int_dct[sequence_lst[i]]
        
    seq = np.asarray(sequence_lst)
    seq = one_hot_encode_X(seq)
    seq = np.transpose(seq)
    return seq

def get_x_balance(cell_type,chromosome,site,genome_distance,strand,histone_modification=True):
    if tempData.cell_type != cell_type:
        tempData.set(cell_type)
    histone_modification = tempData.histone_modification
    
    seq = genome[chromosome][site-genome_distance:site+genome_distance]
    
    

    if strand=="+":
        sequence_lst = list(seq)
    elif strand=="-":
        sequence_lst = reverse_sequence_lst(seq)
    else:
        print("error strand")
        return None
    seq = encode_sequence(sequence_lst)
    
    if histone_modification:
        histone_mark_lst = []
        for i in histone_type_lst:
            one_histone = histone_modification[i].values(chromosome,site-genome_distance, site+genome_distance)
            histone_mark_lst.append(one_histone)

        if strand=="-":
            histone_mark_lst = reverse_histone_mark_lst(histone_mark_lst)
                
        histone_mark = np.asarray(histone_mark_lst)
        histone_mark = np.where(histone_mark < 4, histone_mark, 4)
        # histone_mark = np.log10(histone_mark)*-1

        X = np.concatenate((histone_mark, seq), axis = 0)
        return X
    else:
        return seq
    
def get_original_seq(chromosome,site,genome_distance,strand):

    seq = genome[chromosome][site-genome_distance:site+genome_distance]

    if strand=="+":
        return seq
    elif strand=="-":
        sequence_lst = reverse_sequence_lst(seq)
        return "".join(sequence_lst)
    else:
        print("error strand")
        return None
def get_seq(chromosome,site,genome_distance,strand):
    seq = get_original_seq(chromosome,site,genome_distance,strand)
    seq = ' '.join(list(seq.upper().replace("U", "T"))) # U -> T and add whitespace
    input_ids = tokenizer.encode(seq) # warning: a [CLS] and a [SEP] token will be added to the start and the end of seq
    return input_ids


    

