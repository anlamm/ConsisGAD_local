#!/bin/bash

cd ..


[ -d logs ] || mkdir logs

# python -u main.py --config config/amazon.yml --runs 5 2>&1 | tee logs/train_amazon.log
# python -u main.py --config config/yelp.yml --runs 5 2>&1 | tee logs/train_yelp.log
# python -u main.py --config config/tfinance.yml --runs 5 2>&1 | tee logs/train_tfinance.log


# ###### Remove consistent training, only to validate the GNN encoder
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5  2>&1 | tee logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5  2>&1 | tee logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5  2>&1 | tee logs/train_tfinance.log



# ###### Remove consistent training, only to validate the GNN encoder
# ###### Train on multiple datasets
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_merge.log   #### Use amazon's config

# ###### Then finetune on target dataset
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset amazon 2>&1 | tee logs/finetune_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset yelp 2>&1   | tee logs/finetune_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset tfinance 2>&1  | tee logs/finetune_tfinance.log



###### Remove consistent training, only to validate the GNN encoder
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --random_feature 2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --random_feature  2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --random_feature  2>&1 | tee -a logs/train_tfinance.log



###### Remove consistent training, only to validate the GNN encoder
###### Train on multiple datasets
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 2>&1 --random_feature | tee -a logs/train_merge.log   #### Use amazon's config

###### Then finetune on target dataset
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset amazon --random_feature 2>&1 | tee -a logs/finetune_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset yelp --random_feature 2>&1   | tee -a logs/finetune_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset tfinance --random_feature 2>&1  | tee -a logs/finetune_tfinance.log




###### Remove consistent training, only to validate the GNN encoder
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --drop_edges 2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --drop_edges  2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --drop_edges  2>&1 | tee -a logs/train_tfinance.log



# ###### Train on multiple datasets
CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 --drop_edges 2>&1 | tee -a logs/train_merge.log   #### Use amazon's config
CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/yelp.yml --runs 5 --drop_edges 2>&1 | tee -a logs/train_merge.log   #### Use yelp's config


#### For testing / debugging
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 --random_feature   #### Use amazon's config
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --random_feature 
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --random_feature 
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --random_feature 


cd -