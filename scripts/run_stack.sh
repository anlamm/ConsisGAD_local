#!/bin/bash

cd ..



[ -d logs ] || mkdir logs

# python -u main.py --config config/amazon.yml --runs 5 2>&1 | tee logs/train_amazon.log
# python -u main.py --config config/yelp.yml --runs 5 2>&1 | tee logs/train_yelp.log
# python -u main.py --config config/tfinance.yml --runs 5 2>&1 | tee logs/train_tfinance.log


###### Stack multi layers of ConsisGNN
###### Remove consistent training, only to validate the GNN encoder
##### S1: 2 layers
echo "Stack 2 layers"
CUDA_VISIBLE_DEVICES="5" python -u stack.py --config config/amazon.yml --runs 5 --layers 2 2>&1 | tee -a logs/train_amazon_stack.log
# CUDA_VISIBLE_DEVICES="5" python -u stack.py --config config/yelp.yml --runs 5 --layers 2 2>&1 | tee -a logs/train_yelp_stack.log
# CUDA_VISIBLE_DEVICES="5" python -u stack.py --config config/tfinance.yml --runs 5 --layers 2 2>&1 | tee -a logs/train_tfinance_stack.log

##### S2: 3 layers
echo "Stack 3 layers"
CUDA_VISIBLE_DEVICES="5" python -u stack.py --config config/amazon.yml --runs 5 --layers 3 2>&1 | tee -a logs/train_amazon_stack.log
# CUDA_VISIBLE_DEVICES="5" python -u stack.py --config config/yelp.yml --runs 5 --layers 3 2>&1 | tee -a logs/train_yelp_stack.log
# CUDA_VISIBLE_DEVICES="5" python -u stack.py --config config/tfinance.yml --runs 5 --layers 3 2>&1 | tee -a logs/train_tfinance_stack.log

# python -u baseline.py --config config/yelp.yml --runs 1


cd -