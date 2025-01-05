#!/bin/bash

cd ..


[ -d logs ] || mkdir logs

##### download tfinance and unzip to data/
##### download_data.sh

##### merge multiple datasets and generate train/val/test splits
CUDA_VISIBLE_DEVICES="5" python -u modules/data_loader.py 2>&1 | tee -a logs/offline_split.log

##### pretrain on multiple datasets
CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 1 2>&1 | tee -a logs/train_merge.log   #### Use amazon's config

##### Then finetune on target dataset
CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 1 --load_model --finetune_dataset amazon 2>&1 | tee -a logs/finetune_amazon.log
CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 1 --load_model --finetune_dataset yelp 2>&1   | tee -a logs/finetune_yelp.log
CUDA_VISIBLE_DEVICES="5" python -u finetune.py --config config/amazon.yml --runs 1 --load_model --finetune_dataset tfinance 2>&1  | tee -a logs/finetune_tfinance.log

cd -