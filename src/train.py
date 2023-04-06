
from pytorch_lightning.callbacks import TQDMProgressBar
from lstm_splicing_model import Lightning_module, Multi_site_model, Single_site_model
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from dataset import Single_site_module, Multi_site_module
from args import args
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    
    pl.seed_everything(42)

    logger = TensorBoardLogger('lightning_logs',name="lstm_splicing")

    

    # init model
    model= None

    if args.model=="single":
        
        # W = [11,11,11,11,11,11,11]
        # AR = [1,1,1,1,1,1,1]
        # # in_channels = [32,32,32,32,64,128,256]
        # # out_channels = [32,32,32,64,128,256,512]
        # # in_channels = [16,16,16,16,32,64,128]
        # # out_channels = [16,16,16,32,64,128,256]
        # # in_channels = [8,8,8,8,16,32,64]
        # # out_channels = [8,8,8,16,32,64,128]
        
        
        # # W = [7,7,7,7,7,7,7,7]
        # # AR = [1,1,1,1,1,1,1,1]
        # # in_channels = [32,32,32,32,32,64,128,256]
        # # out_channels = [32,32,32,32,64,128,256,512]
        
        # # W = [11,11,11,11,11,11,11,11]
        # # AR = [1,1,1,1,4,4,4,4]
        # # in_channels = [32,32,32,32,32,32,32,32]
        # # out_channels = [32,32,32,32,32,32,32,32]
        # in_channels = [64,64,64,64,64,64,64,64]
        # out_channels = [64,64,64,64,64,64,64,64]

        model = Single_site_model(512,args.input_channel,args.hidden_size,num_layers=3 ,dropout=args.dropout)
        data_module = Single_site_module(data_dir = args.data_path,batch_size = args.batch_size,num_workers = args.num_workers)
        
    if args.model=="multi":
        model = Multi_site_model(512,args.input_channel,args.hidden_size,num_layers=3 ,dropout=args.dropout)
        data_module = Multi_site_module(data_dir = args.data_path,batch_size = args.batch_size,num_workers = args.num_workers)
        
    
    print(model.parameters())
    transformer = Lightning_module(model,args.task,args.model)

    # if args.load_checkpoint!=None:
    #     print("load model from "+ args.load_checkpoint)
    #     # if args.model=="mix":
    #     #     transformer = Transformer_mix.load_from_checkpoint(args.load_checkpoint,num_layers = 8)
    #     # else:
    #     transformer = Transformer_model.load_from_checkpoint(args.load_checkpoint,num_layers = 8)


    print("----------using {}------------".format(args.device))
    
    trainer = pl.Trainer(accelerator=args.device,val_check_interval= 0.5,default_root_dir=args.checkpoint_dir,logger=logger,max_epochs=args.max_epochs,callbacks=[TQDMProgressBar(refresh_rate=50)])
    # trainer = pl.Trainer(accelerator=args.device,precision='bf16',val_check_interval= 0.01)
    
    trainer.fit(model=transformer,datamodule=data_module)


if __name__=='__main__':
    torch.set_default_dtype(torch.float32)
    print(args)
    main()

