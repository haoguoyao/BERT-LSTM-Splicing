import argparse



def get_parser():
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_shuffle", action="store_true", help="skip files that exist")
    parser.add_argument("--read_out", action="store_true",default=False)
    
    parser.add_argument("--attn_close", action="store_true",default=False)
    parser.add_argument("--relative_position", action="store_true",default=False)
    parser.add_argument("--absolute_position", action="store_true",default=False)
    # For the mix structure
    # parser.add_argument("--multi_model", action="store_true",default=False)
    parser.add_argument('--histone', default='all', choices=['core', 'all','none'])
    parser.add_argument('--mode', default='main', choices=['main', 'ray_tune'])

    parser.add_argument("--exclude_xy", action="store_true",default=False)
    parser.add_argument('--model', choices=['multi','single'])
    parser.add_argument('--single_site_type', default='RNN', choices=['RNN', 'SpliceBERT'])

    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--precision', default='32', choices=['32', 'bf16'])
    parser.add_argument('--reshape_back', default='repeat', choices=['repeat', 'reshape'])
    parser.add_argument('--task', choices=['reg', 'cls'])
    parser.add_argument("--outer_rnn_size",  type=int,default=512)
    #hiddensize = 32
    parser.add_argument("--hidden_size",  type=int,default=512)
    parser.add_argument("--input_channel",  type=int,default=19)
    parser.add_argument("--max_epochs",  type=int,default=100)
    parser.add_argument("--batch_size",  type=int,default=256)
    parser.add_argument("--num_head",  type=int,default=1)
    parser.add_argument("--num_workers",  type=int,default=16)
    parser.add_argument("--learning_rate",  type=float,default=0.0001)
    parser.add_argument("--checkpoint_dir",  type=str,default="./checkpoints")
    parser.add_argument("--load_checkpoint",  type=str,default=None)
    parser.add_argument("--dropout",  type=float,default=0.3)


    parser.add_argument("--patch_num",  type=int,default=256)
    parser.add_argument("--CL",  type=int,default=8192)

    #context length
    parser.add_argument("--CL_max",  type=int,default=8192)
    parser.add_argument('--data_path', action='append', help='<Required> Set flag', required=False)

    
    # parser.add_argument("--data_path",type=str,default="/rhome/ghao004/bigdata/lstm_splicing/single_site_dataset_plus/")
    # parser.add_argument("--data_path",type=str,default="/rhome/ghao004/bigdata/lstm_splicing/single_site_dataset_512_classify_correct2/")
    



    
    
    return parser



parser = get_parser()
args, unknown = parser.parse_known_args()
if args.task == 'reg':
    args.dropout = 0.0



