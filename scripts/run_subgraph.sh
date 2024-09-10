#!/bin/bash

cd ..


[ -d logs ] || mkdir logs

CUDA_VISIBLE_DEVICES="4" python -u subgraph_classification.py --config config/amazon.yml --runs 3 --ego 2>&1 | tee -a logs/train_amazon_subgraph.log
# CUDA_VISIBLE_DEVICES="4" python -u subgraph_classification.py --config config/yelp.yml --runs 3 --ego 2>&1 | tee -a logs/train_yelp_subgraph.log
# CUDA_VISIBLE_DEVICES="4" python -u subgraph_classification.py --config config/tfinance.yml --runs 3 --ego 2>&1 | tee -a logs/train_tfinance_subgraph.log

cd -