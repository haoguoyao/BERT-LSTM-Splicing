import argparse
import numpy as np
import os
from train_val_partition import Train_Chromes, Valid_Chromes, Test_Chromes
from generate_x import get_x_balance,get_seq
from generate_y import get_y, Train_Chromes, Valid_Chromes, Test_Chromes, get_sse_by_gene_id
import numba as nb
import pandas as pd
from tqdm import tqdm
from multiprocessing import Lock,Pool
import ast


def get_gene_dict():

    with open("/rhome/ghao004/bigdata/lstm_splicing/genome/gencode.v42.primary_assembly.annotation.gtf",'r') as g:
        gene_dict = {}
        exon_dict = {}
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



                    if gene_type=="protein_coding" and transcript_type=="protein_coding":

                        if transcript_name in exon_dict:
                            exon_dict[transcript_name].append((chromosome,start-1,end,gene_name,exon_id,strand))
                        else:
                            exon_dict[transcript_name] = [(chromosome,start-1,end,gene_name,exon_id,strand)]
        
        for tx in exon_dict:
            if len(exon_dict[tx])>2:
                transcript=exon_dict[tx]

                # remove first and last exon
                for exon in transcript[1:-1]:
                    gene_id = exon[3]
                    if gene_id in gene_dict:
                        gene_dict[gene_id].append((exon[0],exon[1],exon[2],exon[3],exon[4],exon[5]))
                    else:
                        gene_dict[gene_id] = [(exon[0],exon[1],exon[2],exon[3],exon[4],exon[5])]

        # this gene_dict may have duplicate sites, because the site could appear in different transcripts.
        return gene_dict



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

        if os.path.exists("/rhome/ghao004/bigdata/lstm_splicing/data/{}".format(dataset_name)):
            import shutil
            dir_path = "/rhome/ghao004/bigdata/lstm_splicing/data/{}".format(dataset_name)
            try:
                shutil.rmtree(dir_path)
            except:
                print("An exception occurred")    
        folder_name_dct = {}
        for i in Train_Chromes:
            folder_name_dct[i]="/rhome/ghao004/bigdata/lstm_splicing/data/{}/train/".format(dataset_name)+i
            folder_name_dct[i+"_num"]=0
        for i in Valid_Chromes:
            folder_name_dct[i]="/rhome/ghao004/bigdata/lstm_splicing/data/{}/valid/".format(dataset_name)+i
            folder_name_dct[i+"_num"]=0
        for i in Test_Chromes:
            folder_name_dct[i]="/rhome/ghao004/bigdata/lstm_splicing/data/{}/test/".format(dataset_name)+i
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
    try:
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
        print("a")
        
        #chromosome,start,end,gene_id,exon_id,strand
        for exon in one_gene:
            ystart = exon[1]

            #shift because the annoation and the rna-seq differ by one base
            # if args.task=='reg':
            #     ystart = exon[1]-1
            if exon[1] not in site_list:
                histone_mark,DNA_seq = get_x_balance(args.cell_type,chromosome,exon[1],args.context_length,strand)
                splicing_site1 = {"histone_mark":histone_mark,"DNA_seq":DNA_seq,"y":get_y(args.cell_type,chromosome,ystart,strand,args.task),"position":exon[1],"strand":strand,"raw_seq":get_seq(chromosome,exon[1],255,strand)}
                data_list.append(splicing_site1)
                site_list.append(exon[1])
            if exon[2] not in site_list:
                histone_mark,DNA_seq = get_x_balance(args.cell_type,chromosome,exon[2],args.context_length,strand)
                splicing_site2 = {"histone_mark":histone_mark,"DNA_seq":DNA_seq,"y":get_y(args.cell_type,chromosome,exon[2],strand,args.task),"position":exon[2],"strand":strand,"raw_seq":get_seq(chromosome,exon[2],255,strand)}
                data_list.append(splicing_site2)
                site_list.append(exon[2])
                
        print("a1")
        # gene_from_sse_file = get_sse_by_gene_id("GM12878",gene_id)
        # if len(gene_from_sse_file)>0:
        #     print(gene_id+"found in sse file")
        #     for index, row in gene_from_sse_file.iterrows():
        #         strand = row["Strand"]
        #         chromosome = row["Region"]
        #         site = row["Site"]
        #         y = 0
        #         if site not in site_list:
        #             read_count =row["alpha_count"]+row["beta1_count"]+row["beta2Simple_count"]
        #             if read_count>=20:
        #                 y = row["SSE"]
        #             else:
        #                 y = np.NAN
        #             histone_mark,DNA_seq = get_x_balance(args.cell_type,row["Region"],site,args.context_length,row["Strand"])
        #             splicing_site3 = {"histone_mark":histone_mark,"DNA_seq":DNA_seq,"y":y,"position":row["Site"],"strand":row["Strand"],"raw_seq":get_seq(row["Region"],row["Site"],255,row["Strand"])}
        #             data_list.append(splicing_site3)
        #             site_list.append(site)
        print("a2")
        if len(data_list)>args.max_splicing_site_num:
            return None
            
        # sort the splicing site by upstreaming and downstreaming         
        data_list.sort(key=lambda a: a["position"])
        if strand=="-":
            data_list = data_list[::-1]

        data_dct = {"data":data_list,"gene":gene_id,"chromosome":chromosome,"strand":strand}
        return data_dct
    except Exception as e:
        print(e)
        print("error in {}".format(one_gene[0]))
        return None


# @nb.jit()
def convert_from_exon_to_splicing_site(gene_exon_dict):
    dataset_generator_multi = generate_dataset(args.dataset_name+"_multisite")
    dataset_generator_single = generate_dataset(args.dataset_name+"_singlesite")
    # pool = Pool(processes=16,maxtasksperchild=100)

    pool = Pool(processes=8)
    gene_exon_lst = list(gene_exon_dict.values())
    for data_dct in pool.imap_unordered(do_one_gene, gene_exon_lst):
        if data_dct is None:
            continue
        # save_as_singlesite_data(dataset_generator_single,data_dct)
        save_as_multisite_data(dataset_generator_multi,data_dct)
    print("single dataset size is {}".format(dataset_generator_single.dataset_size))
    print("multi dataset size is {}".format(dataset_generator_multi.dataset_size))
        
    return


def save_as_multisite_data(dataset_generator,data_dct):
    histone_mark_lst = []
    DNA_seq_lst = []
    Y_lst = []
    position_lst = []
    seq_lst = []
    
    for splicing_site in data_dct["data"]:
        histone_mark_lst.append(splicing_site["histone_mark"])
        DNA_seq_lst.append(splicing_site["DNA_seq"])
        Y_lst.append(splicing_site["y"])
        position_lst.append(splicing_site["position"])
        seq_lst.append(splicing_site["raw_seq"])
        
    gene_id = data_dct["gene"]
    chromosome = data_dct["chromosome"]
    strand = data_dct["strand"]
    
    dct = {"chr":chromosome, "histone_mark":histone_mark_lst,"DNA_seq":DNA_seq_lst,"Y":Y_lst,"position" :position_lst,"strand":strand,"gene_id":gene_id,"splicing_site_num":len(Y_lst),"raw_seq":seq_lst}
    
    path = dataset_generator.save_npz(dct)
    if gene_id == "ENSG00000257923.12":
        print("CUX1")
        # print(dct)
        print(path)
    return


def save_as_singlesite_data(dataset_generator,data_dct):
    strand = data_dct["strand"]
    for splicing_site in data_dct["data"]:
        # X_lst.append(splicing_site[0])
        # Y_lst.append(splicing_site[1])
        # position_lst.append(splicing_site[2])
        chromosome = data_dct["chromosome"]
        dct = {"chr":chromosome,"strand":strand,"histone_mark":splicing_site["histone_mark"],"DNA_seq":splicing_site["DNA_seq"],"Y":splicing_site["y"],"raw_seq":splicing_site["raw_seq"]}
        dataset_generator.save_npz(dct)
    return 
        
            
def get_args():
    parser = argparse.ArgumentParser(description='generate multisite dataset')
    parser.add_argument("--max_splicing_site_num",  type=int,default=512)
    parser.add_argument("--context_length",  type=int,default=256)
    parser.add_argument("--cell_type",  type=str,default="GM12878")
    parser.add_argument("--dataset_prefix",  type=str,default="mixed_dataset")
    
    args = parser.parse_args()
    return args

args = get_args()

def generate_val_data():
    import pandas as pd
    args.dataset_name = args.dataset_prefix+"_"+args.cell_type+"_maxsite"+str(args.max_splicing_site_num)+"_context"+str(args.context_length)
    sse_file_url= '/rhome/ghao004/bigdata/lstm_splicing/process_data/bams/GM12878.filtered.SpliSER.tsv'
    sse_file = pd.read_csv(sse_file_url,sep='\t',dtype={"Gene":str})
    sse_file = sse_file.loc[(sse_file['Region'].isin(Train_Chromes+Valid_Chromes)) & (sse_file['Strand'].isin(["+","-"]))]
    sse_file = sse_file.dropna(subset=['Gene'])
    print(sse_file["Gene"].unique())
    print(len(sse_file["Gene"].unique()))


    dataset_generator_multi = generate_dataset(args.dataset_name+"_multisite_val")
    dataset_generator_single = generate_dataset(args.dataset_name+"_singlesite_val")


    for gene_name in sse_file["Gene"].unique():
        this_gene = sse_file.loc[sse_file["Gene"]==gene_name]
        data_list = []
        strand = None

        for index, row in this_gene.iterrows():
            strand = row["Strand"]
            chromosome = row["Region"]
            # partners = row["Partners"]
            site = row["Site"]
            # dct = list(ast.literal_eval(partners).keys())
            # # print(dct)
            # if int(site)<int(dct[0]):
            #     site = site-1

            read_count =row["alpha_count"]+row["beta1_count"]+row["beta2Simple_count"]
            if read_count>=20:
                y = row["SSE"]
            else:
                y = np.NAN
            histone_mark,DNA_seq = get_x_balance(args.cell_type,row["Region"],site,args.context_length,row["Strand"])
            splicing_site = {"histone_mark":histone_mark,"DNA_seq":DNA_seq,"y":y,"position":row["Site"],"strand":row["Strand"],"raw_seq":get_seq(row["Region"],row["Site"],255,row["Strand"])}
            data_list.append(splicing_site)

        if strand=="-":
            data_list = data_list[::-1]


        data_dct = {"data":data_list,"gene":gene_name,"chromosome":chromosome,"strand":strand}
        save_as_singlesite_data(dataset_generator_single,data_dct)
        save_as_multisite_data(dataset_generator_multi,data_dct)

    print("single dataset size is {}".format(dataset_generator_single.dataset_size))
    print("multi dataset size is {}".format(dataset_generator_multi.dataset_size))

# if __name__=="__main__":   
#     generate_val_data()
if __name__=="__main__":         
    

    # filtered_gene_dict = get_gene_dict()

    # save_gene_dict(filtered_gene_dict,"gene_dict_filtered.npy")
    filtered_gene_dict = np.load("gene_dict_filtered.npy",allow_pickle=True).item()


    print("gene_dict_size {}".format(len(filtered_gene_dict)))


    # args.task = "cls"
    # args.dataset_name = args.dataset_prefix+"_"+args.cell_type+"_"+args.task+"_"+str(args.context_length)+"_maxsite"+str(args.max_splicing_site_num)
    # convert_from_exon_to_splicing_site(filtered_gene_dict)


    args.task = "reg"
    args.dataset_name = args.dataset_prefix+"_"+args.cell_type+"_"+args.task+"_"+str(args.context_length)+"_maxsite"+str(args.max_splicing_site_num)
    convert_from_exon_to_splicing_site(filtered_gene_dict)


    print("done")
