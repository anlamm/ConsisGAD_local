#!/bin/bash

cd ..


[ -d logs ] || mkdir logs

# python -u main.py --config config/amazon.yml --runs 5 2>&1 | tee logs/train_amazon.log
# python -u main.py --config config/yelp.yml --runs 5 2>&1 | tee logs/train_yelp.log
# python -u main.py --config config/tfinance.yml --runs 5 2>&1 | tee logs/train_tfinance.log


# ###### Remove consistent training, only to validate the GNN encoder
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5  2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5   2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5   2>&1 | tee -a logs/train_tfinance.log

# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 3 --ego  2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 3   --ego  2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 3   --ego  2>&1 | tee -a logs/train_tfinance.log

# ###### Remove consistent training, only to validate the GNN encoder
# ###### Train on multiple datasets
CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5   2>&1 | tee -a logs/train_merge.log   #### Use amazon's config
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 3 --ego  2>&1 | tee -a logs/train_merge.log   #### Use amazon's config

# ###### Then finetune on target dataset
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset amazon 2>&1 | tee logs/finetune_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset yelp 2>&1   | tee logs/finetune_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 5 --load_model --finetune_dataset tfinance 2>&1  | tee logs/finetune_tfinance.log



# ##### Remove consistent training, only to validate the GNN encoder
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --random_feature 2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --random_feature  2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --random_feature  2>&1 | tee -a logs/train_tfinance.log

# ##### Remove consistent training, only to validate the GNN encoder
# ##### Train on multiple datasets
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
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 --drop_edges 2>&1 | tee -a logs/train_merge.log   #### Use amazon's config
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/yelp.yml --runs 5 --drop_edges 2>&1 | tee -a logs/train_merge.log   #### Use yelp's config



##### Remove consistent training, only to validate the GNN encoder
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --structural_feature 2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --structural_feature  2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --structural_feature  2>&1 | tee -a logs/train_tfinance.log

# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 3 --ego --structural_feature 2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 3 --ego --structural_feature  2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 3 --ego --structural_feature  2>&1 | tee -a logs/train_tfinance.log


##### Remove consistent training, only to validate the GNN encoder
##### Train on multiple datasets
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 2>&1 --structural_feature | tee -a logs/train_merge.log   #### Use amazon's config
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 3 2>&1 --ego --structural_feature | tee -a logs/train_merge.log   #### Use amazon's config

# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --structural_feature  --drop_edges 2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --structural_feature   --drop_edges 2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --structural_feature   --drop_edges 2>&1 | tee -a logs/train_tfinance.log

# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 2>&1 --structural_feature --drop_edges | tee -a logs/train_merge.log   #### Use amazon's config


# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --structural_feature  --drop_edges --cat_feature 2>&1 | tee -a logs/train_amazon.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --structural_feature   --drop_edges  --cat_feature 2>&1 | tee -a logs/train_yelp.log
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --structural_feature   --drop_edges  --cat_feature 2>&1 | tee -a logs/train_tfinance.log

# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 2>&1 --structural_feature --drop_edges  --cat_feature | tee -a logs/train_merge.log   #### Use amazon's config


### Change valid loader from Avg. on 3 datasets to Yelp's valid loader
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_merge.log   #### Use amazon's config, Yelp's valid loader




#### 
# for DATASET in weibo reddit tolokers amazon tfinance yelp questions elliptic
# do
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/$DATASET.yml --runs 5 2>&1 | tee -a logs/train_$DATASET.log
# done

# for DATASET in weibo reddit tolokers amazon tfinance yelp questions elliptic
# do
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/$DATASET.yml --runs 5 --drop_edges 2>&1 | tee -a logs/train_$DATASET.log
# done

# for DATASET in dgraphfin tsocial
# do
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/$DATASET.yml --runs 5 2>&1 | tee -a logs/train_$DATASET.log
# done

# for DATASET in dgraphfin tsocial
# do
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/$DATASET.yml --runs 5 --drop_edges 2>&1 | tee -a logs/train_$DATASET.log
# done






#### For testing / debugging
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 --random_feature   #### Use amazon's config
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 1 --add_edge_feature
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --random_feature 
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --random_feature 
# CUDA_VISIBLE_DEVICES="5" CUDA_LAUNCH_BLOCKING=1 python -u baseline.py --config config/weibo.yml --runs 1 
# CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/weibo.yml --runs 1


cd -