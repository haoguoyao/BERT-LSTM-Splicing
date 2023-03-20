import argparse
import numpy as np
import os
from generate_x import get_x_balance,get_seq
from generate_y import get_y, Train_Chromes, Valid_Chromes, Test_Chromes
import numba as nb

from tqdm import tqdm
from multiprocessing import Lock,Pool



def get_gene_dict():
    with open("/rhome/ghao004/bigdata/lstm_splicing/genome/gencode.v42.primary_assembly.annotation.gtf",'r') as g:
        gene_dict = {}
        for idx,line in enumerate(g.readlines()):
            if line.startswith("#"):
                continue
            else:

                col=line.strip().split("\t")
                chromosome=col[0]
                category=col[2]
                start=int(col[3])
                end=int(col[4])
                strand=col[6]
                extra=col[8].split(";")

                if category=="exon":

                    gene_id=extra[0].strip().split(" ")[1].strip("\"")
                    transcript_id=extra[1].strip().split(" ")[1].strip("\"")

                    gene_type=extra[2].strip().split(" ")[1].strip("\"")
                    gene_name=extra[3].strip().split(" ")[1].strip("\"")

                    transcript_type=extra[4].strip().split(" ")[1].strip("\"")
                    transcript_name=extra[5].strip().split(" ")[1].strip("\"")

                    exon_id=extra[7].strip().split(" ")[1].strip("\"")
                    level=extra[8].strip()


                    if gene_type=="protein_coding" and transcript_type=="protein_coding" and level!="level 3":
                        

                        if gene_id in gene_dict:
                            gene_dict[gene_id].append((chromosome,start,end,gene_id,exon_id,strand))
                        else:
                            gene_dict[gene_id] = [(chromosome,start,end,gene_id,exon_id,strand)]

        print("After loading, gene_dict size {}".format(len(gene_dict)))
        return gene_dict    


#filter by the exon number of a gene
def filter_gene_dict(gene_dict):
                            
    filtered_gene_dict = {}
    for tx in gene_dict:
        if len(gene_dict[tx])>2:
            filtered_gene_dict[tx] = gene_dict[tx]
    print("After filtering, gene_dict size {}".format(len(filtered_gene_dict)))
    return filtered_gene_dict




def save_gene_dict(gene_dict,filename):
    
    np.save(filename, gene_dict)
    return 


class generate_dataset:
    def __init__(self,dataset_name):
        self.dataset_name = dataset_name
        self.generate_dataset_structure(self.dataset_name)
        self.mutex = Lock()
        self.dataset_size = 0
        
    def generate_dataset_structure(self,dataset_name):

        if os.path.exists("/rhome/ghao004/bigdata/lstm_splicing/{}".format(dataset_name)):
            import shutil
            dir_path = "/rhome/ghao004/bigdata/lstm_splicing/{}".format(dataset_name)
            try:
                shutil.rmtree(dir_path)
            except:
                print("An exception occurred")    
        folder_name_dct = {}
        for i in Train_Chromes:
            folder_name_dct[i]="/rhome/ghao004/bigdata/lstm_splicing/{}/train/".format(dataset_name)+i
            folder_name_dct[i+"_num"]=0
        for i in Valid_Chromes:
            folder_name_dct[i]="/rhome/ghao004/bigdata/lstm_splicing/{}/valid/".format(dataset_name)+i
            folder_name_dct[i+"_num"]=0
        for i in Test_Chromes:
            folder_name_dct[i]="/rhome/ghao004/bigdata/lstm_splicing/{}/test/".format(dataset_name)+i
            folder_name_dct[i+"_num"]=0

        for i in Train_Chromes+Valid_Chromes+Test_Chromes:
            if not os.path.exists(folder_name_dct[i]):
                os.makedirs(folder_name_dct[i])
        self.folder_name_dct = folder_name_dct 


    def save_file_get_path(self,_chr):
        self.mutex.acquire()
        self.dataset_size+=1
        _return = self.folder_name_dct[_chr]+'/'+str(self.folder_name_dct[_chr+"_num"])
        self.folder_name_dct[_chr+"_num"]+=1
        self.mutex.release()
        if self.folder_name_dct[_chr+"_num"]%100 == 0:
            print(_return)
        return _return

    

    def save_npz(self,dct):
        if dct['chr'] in Train_Chromes+Valid_Chromes+Test_Chromes:
            path = self.save_file_get_path(dct['chr'])
            np.savez(path,**dct)
            os.chmod(path+".npz", 0o771)
        return path
        


def do_one_gene(one_gene):
    strand = one_gene[0][5]
    if strand not in ["+","-"]:
        print("strand information error")
        print("strand is {}".format(strand))
        return
    
    chromosome = one_gene[0][0]
    gene_id = one_gene[0][3]
    #to remove duplication
    site_list = []
    data_list = []
    
    #chromosome,start,end,gene_id,exon_id,strand
    for exon in one_gene:
        ystart = exon[1]
        if args.task=='reg':
            ystart = exon[1]-1
        if exon[1] not in site_list:
            splicing_site1 = (get_x_balance(args.cell_type,chromosome,exon[1],args.context_length,strand),get_y(args.cell_type,chromosome,ystart,strand,args.task),exon[1],strand,get_seq(chromosome,exon[1],255,strand))
            data_list.append(splicing_site1)
            site_list.append(exon[1])
        if exon[2] not in site_list:
            splicing_site2 = (get_x_balance(args.cell_type,chromosome,exon[2],args.context_length,strand),get_y(args.cell_type,chromosome,exon[2],strand,args.task),exon[2],strand,get_seq(chromosome,exon[2],255,strand))
            data_list.append(splicing_site2)
            site_list.append(exon[2])
            

    if len(data_list)>args.max_splicing_site_num:
        return None
        
    # sort the splicing site by upstreaming and downstreaming         
    data_list.sort(key=lambda a: a[2])
    if strand=="-":
        data_list = data_list[::-1]

    data_dct = {"data":data_list,"gene":gene_id,"chromosome":chromosome,"strand":strand}
    return data_dct


# @nb.jit()
def convert_from_exon_to_splicing_site(gene_exon_dict):
    dataset_generator_multi = generate_dataset(args.dataset_name+"_multisite")
    dataset_generator_single = generate_dataset(args.dataset_name+"_singlesite")
    pool = Pool(processes=32)
    gene_exon_lst = list(gene_exon_dict.values())
    for data_dct in pool.imap_unordered(do_one_gene, gene_exon_lst):
        if data_dct is None:
            continue
        save_as_singlesite_data(dataset_generator_single,data_dct)
        save_as_multisite_data(dataset_generator_multi,data_dct)
    print("single dataset size is {}".format(dataset_generator_single.dataset_size))
    print("multi dataset size is {}".format(dataset_generator_multi.dataset_size))
        
    return


def save_as_multisite_data(dataset_generator,data_dct):
    X_lst = []
    Y_lst = []
    position_lst = []
    seq_lst = []
    
    for splicing_site in data_dct["data"]:
        X_lst.append(splicing_site[0])
        Y_lst.append(splicing_site[1])
        position_lst.append(splicing_site[2])
        seq_lst.append(splicing_site[4])
        
    gene_id = data_dct["gene"]
    chromosome = data_dct["chromosome"]
    strand = data_dct["strand"]
    
    dct = {"chr":chromosome, "X":X_lst,"Y":Y_lst,"position" :position_lst,"strand":strand,"gene_id":gene_id,"splicing_site_num":len(Y_lst),"seq":seq_lst}
    
    path = dataset_generator.save_npz(dct)
    if gene_id == "ENSG00000257923.12":
        print("CUX1")
        # print(dct)
        print(path)
    return


def save_as_singlesite_data(dataset_generator,data_dct):
    for splicing_site in data_dct["data"]:
        # X_lst.append(splicing_site[0])
        # Y_lst.append(splicing_site[1])
        # position_lst.append(splicing_site[2])
        chromosome = data_dct["chromosome"]
        dct = {"chr":chromosome,"X":splicing_site[0],"Y":splicing_site[1],"seq":splicing_site[4]}
        dataset_generator.save_npz(dct)
    return 
        
            
def get_args():
    parser = argparse.ArgumentParser(description='generate multisite dataset')
    parser.add_argument("--remove_first_and_last",  type=int,default=0)
    parser.add_argument("--max_splicing_site_num",  type=int,default=512)
    parser.add_argument("--context_length",  type=int,default=256)
    parser.add_argument("--cell_type",  type=str,default="GM12878")
    parser.add_argument("--dataset_prefix",  type=str,default="best_data_all")
    
    args = parser.parse_args()
    return args

args = get_args()


if __name__=="__main__":         
    

    # gene_dict = get_gene_dict()
    # filtered_gene_dict = filter_gene_dict(gene_dict)
    # save_gene_dict(filtered_gene_dict,"gene_dict.npy")
    filtered_gene_dict = np.load("gene_dict.npy",allow_pickle=True).item()
    
    print("gene_dict_size {}".format(len(filtered_gene_dict)))


    args.task = "cls"
    args.dataset_name = args.dataset_prefix+"_"+args.cell_type+"_"+args.task+"_"+str(args.context_length)+"_maxsite"+str(args.max_splicing_site_num)
    convert_from_exon_to_splicing_site(filtered_gene_dict)


    args.task = "reg"
    args.dataset_name = args.dataset_prefix+"_"+args.cell_type+"_"+args.task+"_"+str(args.context_length)+"_maxsite"+str(args.max_splicing_site_num)
    convert_from_exon_to_splicing_site(filtered_gene_dict)


    print("done")
