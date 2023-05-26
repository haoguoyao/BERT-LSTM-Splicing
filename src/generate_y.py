import numpy as np
import pandas as pd
import numba as nb

from train_val_partition import Train_Chromes, Valid_Chromes, Test_Chromes, strands

min_read = 10
class TempData:
    def __init__(self):
        self.cell_type = None
        self.sse_file = None

    def set(self, cell_type):
        self.cell_type = cell_type
        if cell_type == "GM12878":
            
            sse_file_url= '/rhome/ghao004/bigdata/lstm_splicing/process_data/bams/GM12878.filtered.SpliSER.tsv'
            fpkm_file_url= '/rhome/ghao004/bigdata/esprnn/detailed_fpkm.csv'
            self.fpkm_file = pd.read_csv(fpkm_file_url,sep=',')
            self.sse_file = pd.read_csv(sse_file_url,sep='\t')
            self.sse_file = self.sse_file.loc[(self.sse_file['Region'].isin(Train_Chromes+Valid_Chromes+Test_Chromes)) & (self.sse_file['Strand'].isin(strands))]
            print("before filtering sse ",self.sse_file.shape)
            self.sse_file = self.sse_file.dropna(subset=['Gene'])
            print("after filtering sse ",self.sse_file.shape)


        if cell_type=="HepG2":
            sse_file_url= '/rhome/ghao004/bigdata/lstm_splicing/HepG2/bams/HepG2.filtered.SpliSER.tsv'



tempData = TempData()
def get_sse_by_gene_id(cell_type,gene_id):
    print("get sse by gene id")
    if tempData.cell_type != cell_type:
        tempData.set(cell_type)
    sse_file = tempData.sse_file
    sse_row = sse_file.loc[(sse_file['Gene']==gene_id)]
    print("finish sse by gene id")
    return sse_row

def get_y(cell_type,chromosome,site,strand,task):
    if tempData.cell_type != cell_type:
        tempData.set(cell_type)


    if task=="reg":
        sse_file = tempData.sse_file
        sse_row = sse_file.loc[(sse_file['Region']==chromosome)& (sse_file['Site']==site)& (sse_file['Strand']==strand)]

        if (sse_row.shape)[0]==0:
            # print("region",chromosome,"site",site,"strand",strand,"not found in sse file")


            Y = 0
        elif (sse_row.shape)[0]>1:
            print("multiple match, error")
            print(1/0)
            return 
        else:
            read_count = sse_row.iloc[0,:].loc["alpha_count"]+sse_row.iloc[0,:].loc["beta1_count"]+sse_row.iloc[0,:].loc["beta2Simple_count"]
            # print("region",chromosome,"site",site,"strand",strand,"found")

            if read_count>=20:
                Y = sse_row.iloc[0,:].loc["SSE"]
            else:
                # print("region",chromosome,"site",site,"strand",strand,"has less than 20 reads")
                # when read not enough, set sse to nan value
                Y = np.NAN

    elif task=="cls":
        fpkm_filtered1 = tempData.fpkm_file.loc[(tempData.fpkm_file['site_start']==site)|(tempData.fpkm_file['site_end']==site)]
        fpkm_row = fpkm_filtered1.loc[(fpkm_filtered1['chromosome']==chromosome)& (fpkm_filtered1['strand']==strand)]


        if fpkm_row.shape[0]==0:
            Y = 0
        else:
            fpkm_max = fpkm_row["fpkm"].max()
            if fpkm_max>1:
                Y = 1
            else:
                Y = 0
    return Y

