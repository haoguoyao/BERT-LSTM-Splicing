from pytorch_lightning.callbacks import TQDMProgressBar
from lstm_splicing_model import Lightning_module, Multi_site_model, Single_site_model
import os
import torch
import pytorch_lightning as pl
from dataset import Single_site_module, Multi_site_module
from args import args
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback,TuneReportCheckpointCallback
tune.execution.ray_trial_executor.DEFAULT_GET_TIMEOUT = 10000

def train_ray_tune(config):


    logger=TensorBoardLogger(save_dir=os.getcwd(), name="raytune", version="v1"),
    data_module = Single_site_module(data_dir = args.data_path,batch_size = args.batch_size,num_workers = args.num_workers)
    model = Single_site_model(512,args.input_channel,config["hidden_size"],config["num_layers"] ,dropout=config["dropout"],model_type= config["model_type"],prune_ratio = config["prune_ratio"])

    transformer = Lightning_module(model,args.task,args.model,config["learning_rate"])
    
    trainer = pl.Trainer(accelerator=args.device,val_check_interval= 0.5,default_root_dir=args.checkpoint_dir,logger=logger,max_epochs=args.max_epochs,callbacks=[TQDMProgressBar(refresh_rate=200),TuneReportCallback({"loss": "val_loss","F1":"val_F1","AUROC":"val_AUROC","AUPRC":"val_AUPRC","spearman":"val_spearman","pearson":"val_pearson"},on="validation_end")])

    trainer.fit(model=transformer,datamodule=data_module)

def train_ray_tune_multi(config):

    logger=TensorBoardLogger(save_dir=os.getcwd(), name="raytune", version="v1"),
    model = Multi_site_model(512,args.input_channel,config["hidden_size"],num_layers=config["num_layers"],dropout=config["dropout"],outer_hidden_size = config["outer_hidden_size"])
    data_module = Multi_site_module(data_dir = args.data_path,batch_size = args.batch_size,num_workers = args.num_workers)
    transformer = Lightning_module(model,args.task,args.model,config["learning_rate"])
    trainer = pl.Trainer(accelerator=args.device,val_check_interval= 0.5,default_root_dir=args.checkpoint_dir,logger=logger,max_epochs=args.max_epochs,callbacks=[TQDMProgressBar(refresh_rate=200),TuneReportCallback({"loss": "val_loss","F1":"val_F1","AUROC":"val_AUROC","AUPRC":"val_AUPRC","spearman":"val_spearman","pearson":"val_pearson"},on="validation_end")])
    trainer.fit(model=transformer,datamodule=data_module)

def ray_tune_main():
    if args.task=="cls" and args.model=="single" and args.single_site_type!="SpliceBERT":
        config = {
            "hidden_size": tune.choice([16,32,64,128]),
            "model_type":tune.choice(["GRU","LSTM"]),
            "dropout": tune.choice([0, 0.15, 0.3,0.45]),
            "learning_rate": tune.loguniform(1e-4, 5*1e-3),
            "prune_ratio": tune.choice([0]),
            "num_layers": tune.choice([2, 3, 4]),
            "outer_hidden_size": tune.choice([0])
        }
    elif args.task=="cls" and args.model=="multi" and args.single_site_type!="SpliceBERT":
        config = {
            "hidden_size": tune.choice([32]),
            "model_type":tune.choice(["GRU"]),
            "dropout": tune.choice([0.2]),
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "prune_ratio": tune.choice([0]),
            "num_layers": tune.choice([3]),
            "outer_hidden_size": tune.choice([1024, 2048, 4096])
        }
    elif args.task=="reg" and args.model=="multi" and args.single_site_type!="SpliceBERT":
        config = {
            "hidden_size": tune.choice([32,64,128,256]),
            "model_type":tune.choice(["GRU"]),
            "dropout": tune.choice([0]),
            "learning_rate": tune.loguniform(1e-6, 2*1e-4),
            "prune_ratio": tune.choice([0]),
            "num_layers": tune.choice([3, 4, 5]),
            "outer_hidden_size": tune.choice([1024, 2048, 4096])
        }
    elif args.task=="reg" and args.model=="single" and args.single_site_type!="SpliceBERT":
        config = {
            "hidden_size": tune.choice([64,128,256,512]),
            "model_type":tune.choice(["GRU","LSTM"]),
            "dropout": tune.choice([0]),
            "learning_rate": tune.loguniform(1e-4, 5*1e-3),
            "prune_ratio": tune.choice([0]),
            "num_layers": tune.choice([ 4,5,6]),
            "outer_hidden_size": tune.choice([0])
        }



    # splicebert
    elif args.task=="cls" and args.model=="single" and args.single_site_type=="SpliceBERT":
        config = {
            "hidden_size": tune.choice([527]),
            "model_type":tune.choice(["bert"]),
            "dropout": tune.choice([0,0.1,0.2,0.3]),
            "prune_ratio": tune.choice([0.1,0.2,0.3,0.4,0.5]),
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "num_layers": tune.choice([0]),
            "outer_hidden_size": tune.choice([0])
        }
    elif args.task=="reg" and args.model=="single" and args.single_site_type=="SpliceBERT":
        config = {
            "hidden_size": tune.choice([527]),
            "model_type":tune.choice(["bert"]),
            "dropout": tune.choice([0]),
            "prune_ratio": tune.choice([0.6,0.2,0.3,0.4,0.5]),
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "num_layers": tune.choice([0]),
            "outer_hidden_size": tune.choice([0])
        }
    

    scheduler = ASHAScheduler(max_t=50, grace_period=8, reduction_factor=2)
    resources_per_trial = {"cpu": 15, "gpu": 1}
    if args.model=="multi":
        train_fn_with_parameters = tune.with_parameters(train_ray_tune_multi)
    elif args.model=="single":
        train_fn_with_parameters = tune.with_parameters(train_ray_tune)
    reporter = CLIReporter(parameter_columns=["model_type","hidden_size","prune_ratio","outer_hidden_size", "dropout","learning_rate","num_layers"],metric_columns=["loss","F1","AUROC","AUPRC","spearman","pearson"])
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=50,
        ),
        run_config=air.RunConfig(
            local_dir="/rhome/ghao004/bigdata/lstm_splicing/raytune_result",
            name=args.raytune_name,
            progress_reporter=reporter,
        ),
        param_space=config,
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
    return

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
    if args.mode=="ray_tune":
        ray_tune_main()
    else:
        main()