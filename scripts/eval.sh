#!/bin/bash

cd ..


[ -d logs ] || mkdir logs



# ###### Remove consistent training, only to validate the GNN encoder
CUDA_VISIBLE_DEVICES="5" python -u eval.py --config config/amazon.yml --runs 5 --load_model 2>&1 | tee logs/eval_amazon.log
CUDA_VISIBLE_DEVICES="5" python -u eval.py --config config/yelp.yml --runs 5 --load_model  2>&1 | tee logs/eval_yelp.log
CUDA_VISIBLE_DEVICES="5" python -u eval.py --config config/tfinance.yml --runs 5 --load_model  2>&1 | tee logs/eval_tfinance.log
# CUDA_VISIBLE_DEVICES="5" python -u eval.py --config config/amazon.yml --runs 5 --load_model --dataset 'merge' 2>&1 | tee logs/eval_amazon.log

# ###### Remove consistent training, only to validate the GNN encoder
# ###### Train on multiple datasets
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/eval_merge.log   #### Use amazon's config

# ###### Then finetune on target dataset
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset amazon 2>&1 | tee logs/finetune_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset yelp 2>&1   | tee logs/finetune_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset tfinance 2>&1  | tee logs/finetune_tfinance.log








#### For testing / debugging
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 --random_feature   #### Use amazon's config
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --random_feature 
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --random_feature 
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --random_feature 


cd -