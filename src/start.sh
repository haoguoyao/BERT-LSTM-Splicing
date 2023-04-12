
#RNN single
CUDA_VISIBLE_DEVICES=1 python train.py --num_workers 16 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --model="single" --task="reg" 2>&1 | tee log_single_reg_GM12878_removed.txt
CUDA_VISIBLE_DEVICES=2 python train.py --num_workers 16 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --model="single" --task="cls" 2>&1 | tee log_single_cls_GM12878_removed.txt

#ray tune gpu03 single
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --model="single" --task="cls" --raytune_name="tune_cls" --mode="ray_tune" 2>&1 | tee log_raytune_rnn_cls.txt
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 15 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --model="single" --task="reg" --raytune_name="tune_reg" --mode="ray_tune" 2>&1 | tee log_raytune_rnn_reg.txt

#ray tune multi
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 15 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_multisite/" --model="multi" --task="reg" --raytune_name="tune_reg_multi" --mode="ray_tune" 2>&1 | tee log_raytune_multi_reg.txt
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_multisite/" --model="multi" --task="cls" --raytune_name="tune_cls_multi" --mode="ray_tune" 2>&1 | tee log_raytune_multi_cls.txt


#gpu06 single tune splicebert
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --model="single" --task="cls" --raytune_name="tune_cls_bert" --mode="ray_tune" --single_site_type="SpliceBERT" --batch_size=128 2>&1 | tee log_raytune_bert_cls.txt
CUDA_VISIBLE_DEVICES="0,1,2,3" python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --model="single" --task="reg" --raytune_name="tune_reg_bert" --mode="ray_tune" --single_site_type="SpliceBERT" --batch_size=128 2>&1 | tee log_raytune_bert_reg.txt


CUDA_VISIBLE_DEVICES=1 python train.py --num_workers 16 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --hidden_size=527 --histone="all" --model="single" --task="reg" --single_site_type="SpliceBERT" --batch_size=16 2>&1 | tee log_single_reg_GM12878_splicebert_prune75.txt
#Splicebert single
CUDA_VISIBLE_DEVICES=2 python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --hidden_size=527 --histone="all" --model="single" --task="cls" --single_site_type="SpliceBERT" --batch_size=16 2>&1 | tee log_single_cls_GM12878_splicebert.txt
CUDA_VISIBLE_DEVICES=0 python train.py --num_workers 16 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --hidden_size=527 --histone="all" --model="single" --task="reg" --single_site_type="SpliceBERT" --batch_size=128 2>&1 | tee log_single_reg_GM12878_splicebert.txt
CUDA_VISIBLE_DEVICES=4 python train.py --num_workers 8 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/" --hidden_size=512 --histone="none" --model="single" --task="reg" --single_site_type="SpliceBERT" --batch_size=16 2>&1 | tee log_single_reg_GM12878_splicebert_nohistone.txt

CUDA_VISIBLE_DEVICES=4 python train.py --num_workers 16 --data_path "/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_cls_256_maxsite512_singlesite/" --hidden_size=527 --model="single" --task="cls" --single_site_type="SpliceBERT" --batch_size=32 2>&1 | tee log_single_cls_GM12878_splicebert.txt


/rhome/ghao004/bigdata/lstm_splicing/structure_new2_GM12878_reg_256_maxsite512_singlesite/train/chrX/10063.npz