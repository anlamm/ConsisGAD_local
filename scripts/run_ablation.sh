#!/bin/bash

cd ..



[ -d logs ] || mkdir logs

# python -u main.py --config config/amazon.yml --runs 5 2>&1 | tee logs/train_amazon.log
# python -u main.py --config config/yelp.yml --runs 5 2>&1 | tee logs/train_yelp.log
# python -u main.py --config config/tfinance.yml --runs 5 2>&1 | tee logs/train_tfinance.log


###### Ablation of ConsisGNN (Sequence 1)
###### Remove consistent training, only to validate the GNN encoder
##### A1: w/o proj_skip
# echo "Ablation1: w/o proj_skip"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log


#### A2: w/o proj_skip, only aggregate src
# echo "Ablation2: w/o homophily-aware aggregation"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log


#### A3: w/o skip connection
# echo "Ablation3: w/o skip connection"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log




#### A4: w/o proj_in
# echo "Ablation4: w/o proj_in"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log



#### A5: w homophily-aware aggregation
# echo "Ablation5: w homophily-aware aggregation"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log


######################################  分割线 ######################################

###### Ablation of ConsisGNN (Sequence 2)
###### Remove consistent training, only to validate the GNN encoder

#### A6: only aggregate src
# echo "Ablation6: w/o homophily-aware aggregation"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log

#### A6.1: only aggregate dst
echo "Ablation6.1: w/o homophily-aware aggregation"
CUDA_VISIBLE_DEVICES="7" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
CUDA_VISIBLE_DEVICES="7" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
CUDA_VISIBLE_DEVICES="7" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log


### A7: w/o skip connection
# echo "Ablation7: w/o skip connection"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log


#### A8: w homophily-aware aggregation
# echo "Ablation8: w homophily-aware aggregation"
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/amazon.yml --runs 5 2>&1 | tee -a logs/train_amazon_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/yelp.yml --runs 5 2>&1 | tee -a logs/train_yelp_ablation.log
# CUDA_VISIBLE_DEVICES="6" python -u ablation.py --config config/tfinance.yml --runs 5 2>&1 | tee -a logs/train_tfinance_ablation.log




# python -u baseline.py --config config/yelp.yml --runs 1


cd -