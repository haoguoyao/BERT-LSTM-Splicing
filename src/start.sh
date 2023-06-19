
#RNN single
CUDA_VISIBLE_DEVICES="5" python train.py --num_workers 12 --batch_size 32 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --model="single" --task="reg" 2>&1 | tee log_single_reg_GM12878_removed_apr18.txt
CUDA_VISIBLE_DEVICES=2 python train.py --num_workers 16 --batch_size 32 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --model="single" --task="cls" 2>&1 | tee log_single_cls_GM12878_removed.txt

#ray tune gpu03 single
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --model="single" --task="cls" --raytune_name="tune_cls" --mode="ray_tune" 2>&1 | tee log_raytune_rnn_cls.txt
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 15 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --model="single" --task="reg" --raytune_name="tune_reg" --mode="ray_tune" 2>&1 | tee log_raytune_rnn_reg.txt

CUDA_VISIBLE_DEVICES="0" python train.py --num_workers 8 --learning_rate 5e-6 --data_path "/rhome/ghao004/bigdata/lstm_splicing/data/structure_new2_GM12878_reg_256_maxsite512_multisite/" --model="multi" --task="reg" --raytune_name="reg_multi_mask" 2>&1 | tee log_mask_multi_reg.txt

CUDA_VISIBLE_DEVICES="3,4,5" python train.py --num_workers 15 --data_path "/rhome/ghao004/bigdata/lstm_splicing/data/mask_dataset_GM12878_reg_256_maxsite512_singlesite/" --model="single" --task="reg" --raytune_name="tune_reg_multi_attention_for_best" --mode="ray_tune" 2>&1 | tee log_raytune_multi_reg_attention_for_best_apr27.txt
#multi site
CUDA_VISIBLE_DEVICES="4" python train.py --num_workers 16 --learning_rate 5e-6 --data_path "/rhome/ghao004/bigdata/lstm_splicing/data/mask_dataset_GM12878_reg_256_maxsite512_multisite/" --model="multi" --task="reg" --raytune_name="reg_multi_mask" 2>&1 | tee log_mask_multi_reg.txt

CUDA_VISIBLE_DEVICES="0" python train.py --num_workers 16 --learning_rate 5e-6 --data_path "/rhome/ghao004/bigdata/lstm_splicing/data/mask_dataset_GM12878_reg_256_maxsite512_multisite/" --model="multi" --task="reg" --raytune_name="reg_multi_mask" 2>&1 | tee log_mask_multi_reg2.txt
CUDA_VISIBLE_DEVICES="3,4,5" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_multisite/" --model="multi" --task="cls" --raytune_name="tune_cls_multi" --mode="ray_tune" 2>&1 | tee log_raytune_multi_cls.txt

# decide attention
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 15 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_multisite/" --model="multi" --task="reg" --raytune_name="tune_reg_multi_attention3" --mode="ray_tune" 2>&1 | tee log_raytune_multi_reg_attention_3.txt

CUDA_VISIBLE_DEVICES="3,4,5" python train.py --num_workers 15 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_multisite/" --model="multi" --task="reg" --raytune_name="tune_reg_multi_attention_for_best" --mode="ray_tune" 2>&1 | tee log_raytune_multi_reg_attention_for_best_apr27.txt


#gpu06 single tune splicebert
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --model="single" --task="cls" --raytune_name="tune_cls_bert" --mode="ray_tune" --single_site_type="SpliceBERT" --batch_size=128 2>&1 | tee log_raytune_bert_cls.txt
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --model="single" --task="reg" --raytune_name="tune_reg_bert" --mode="ray_tune" --single_site_type="SpliceBERT" --batch_size=128 2>&1 | tee log_raytune_bert_reg.txt


CUDA_VISIBLE_DEVICES=1 python train.py --num_workers 16 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --hidden_size=527 --histone="all" --model="single" --task="reg" --single_site_type="SpliceBERT" --batch_size=16 2>&1 | tee log_single_reg_GM12878_splicebert_prune75.txt


#Splicebert single
CUDA_VISIBLE_DEVICES=2 python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --hidden_size=527 --histone="all" --model="single" --task="cls" --single_site_type="SpliceBERT" --batch_size=16 2>&1 | tee log_single_cls_GM12878_splicebert.txt
CUDA_VISIBLE_DEVICES=6 python train.py --num_workers 4 --batch_size 4 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --hidden_size=527 --histone="all" --model="single" --task="reg" --single_site_type="SpliceBERT"  2>&1 | tee log_single_reg_GM12878_splicebert.txt
CUDA_VISIBLE_DEVICES=4 python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --hidden_size=512 --histone="none" --model="single" --task="reg" --single_site_type="SpliceBERT" --batch_size=16 2>&1 | tee log_single_reg_GM12878_splicebert_nohistone.txt

CUDA_VISIBLE_DEVICES=4 python train.py --num_workers 16 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --hidden_size=527 --model="single" --task="cls" --single_site_type="SpliceBERT" --batch_size=32 2>&1 | tee log_single_cls_GM12878_splicebert.txt




CUDA_VISIBLE_DEVICES=1 python train.py --num_workers 4 --batch_size 1 --learning_rate 2e-6 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --hidden_size=527 --histone="all" --model="multi" --task="reg" --single_site_type="SpliceBERT"  2>&1 | tee log_multi_reg_GM12878_splicebert.txt

/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/train/chrX/10063.npz