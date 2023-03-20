
# from pytorch_lightning.callbacks import TQDMProgressBar
# from lstm_splicing_model import Single_site_lightning, Multi_site_model, Single_site_model
# import os
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import DataLoader, random_split
# import pytorch_lightning as pl
# from dataset import Single_site_module, Multi_site_module
# import argparse
# from pytorch_lightning.loggers import TensorBoardLogger
# import numpy as np
# import matplotlib.pyplot as plt


# def visualize(args):
    
#     pl.seed_everything(42)



#     checkpoint_ckpt = os.listdir(args.checkpoint_dir)[0]
#     checkpoint = torch.load(args.checkpoint_dir+checkpoint_ckpt)
#     # print(checkpoint.keys())
#     multi_module = Multi_site_model(512,19,512,num_layers=3 ,dropout=0)
#     multi_module.attention.relative_position = True

#     model = Single_site_lightning.load_from_checkpoint(args.checkpoint_dir+checkpoint_ckpt,model = multi_module,task = "reg",multi_model = True)
#     model.to("cpu")
#     # disable randomness, dropout, etc...
#     model.eval()

#     npzfile = np.load(args.data_path)
#     x = np.array(npzfile['X'])
#     y = np.array(npzfile['Y'])
#     # seq = npzfile['seq']
#     position = np.array(npzfile['position'])
#     print("position")
#     print(position)
#     # print(1/0)
#     print("y")
#     print(y)
#     site_num= np.array(npzfile["splicing_site_num"])
    
    
#     x = x.astype(np.single)  
#     y = y.astype(np.single)
#     print(x.shape)
#     # print(1/0)
#     # indices = list(range(11))+list(range(-4,0))
#     # x = x[:,indices]
#     position = position.astype(np.single)
#     site_num = site_num.astype(np.single)
#     position = np.absolute(position-position[0])


#     dct =  {"x":x,"y":y,"position":position,"site_num":site_num,"seq":1}
#     y_hat,attention = model.forward_visualization(torch.from_numpy(x),torch.from_numpy(position),1)
#     print("y_hat")
#     print(y_hat.squeeze())
#     print(attention.shape)
#     attention = attention.squeeze().detach().numpy()

#     pyplot.matshow(attention[0])
#     pyplot.savefig("attention0.jpg")
#     pyplot.matshow(attention[1])
#     pyplot.savefig("attention1.jpg")
#     pyplot.matshow(attention[2])
#     pyplot.savefig("attention2.jpg")
#     pyplot.matshow(attention[3])
#     pyplot.savefig("attention3.jpg")
#     pyplot.matshow(attention[0]+attention[1]+attention[2]+attention[3])
#     pyplot.savefig("attention4.jpg")

# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description="Visualize attention weights",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--checkpoint_dir",  type=str,default="./lightning_logs/lstm_splicing/version_567/checkpoints/")
#     parser.add_argument("--data_path",  type=str,default="/rhome/ghao004/bigdata/lstm_splicing/best_data_all_GM12878_reg_256_maxsite512_multisite/test/chr7/595.npz")
#     parser.add_argument("--relative_position", action="store_true",default=False)

#     args = parser.parse_args()
#     args.relative_position = True
#     args.absolute_position = False
#     print(args)
#     visualize(args)




from pytorch_lightning.callbacks import TQDMProgressBar
from lstm_splicing_model import Single_site_lightning, Multi_site_model, Single_site_model
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from dataset import Single_site_module, Multi_site_module
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt


def visualize(args):
    
    pl.seed_everything(42)



    checkpoint_ckpt = os.listdir(args.checkpoint_dir)[0]
    checkpoint = torch.load(args.checkpoint_dir+checkpoint_ckpt)
    # print(checkpoint.keys())
    multi_module = Multi_site_model(512,19,512,num_layers=3 ,dropout=0)
    multi_module.attention.relative_position = True

    model = Single_site_lightning.load_from_checkpoint(args.checkpoint_dir+checkpoint_ckpt,model = multi_module,task = "reg",multi_model = True)
    model.to("cpu")
    # disable randomness, dropout, etc...
    model.eval()

    npzfile = np.load(args.data_path)
    x = np.array(npzfile['X'])
    y = np.array(npzfile['Y'])
    # seq = npzfile['seq']
    position = np.array(npzfile['position'])
    print("position")
    print(position)
    # print(1/0)
    print("y")
    print(y)
    site_num= np.array(npzfile["splicing_site_num"])
    
    
    x = x.astype(np.single)  
    y = y.astype(np.single)
    print(x.shape)
    # print(1/0)
    # indices = list(range(11))+list(range(-4,0))
    # x = x[:,indices]
    position = position.astype(np.single)
    site_num = site_num.astype(np.single)
    position = np.absolute(position-position[0])
    


    dct =  {"x":x,"y":y,"position":position,"site_num":site_num,"seq":1}
    y_hat,attention = model.forward_visualization(torch.from_numpy(x),torch.from_numpy(position),1)
    print("y_hat")
    print(y_hat.squeeze())
    print(attention.shape)
    attention = attention.squeeze().detach().numpy()
    # plt.colorbar()
    for i in range(attention.shape[0]):
        fig = plt.figure()
        matshow = plt.matshow(attention[i])
        plt.colorbar(matshow)

        plt.savefig("attention"+str(i)+".jpg")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Visualize attention weights",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint_dir",  type=str,default="./lightning_logs/lstm_splicing/version_567/checkpoints/")
    parser.add_argument("--data_path",  type=str,default="/rhome/ghao004/bigdata/lstm_splicing/best_data_all_GM12878_reg_256_maxsite512_multisite/test/chr7/595.npz")

    args = parser.parse_args()
    args.relative_position = True
    args.absolute_position = False
    print(args)
    visualize(args)

