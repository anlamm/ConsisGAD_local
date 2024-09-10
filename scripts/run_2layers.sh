#!/bin/bash

cd ..


[ -d logs ] || mkdir logs

for LAYERS in 2 3 4
do
for DATASET in weibo reddit tolokers amazon tfinance yelp questions elliptic
do
CUDA_VISIBLE_DEVICES="6" python -u baseline.py --config config/$DATASET.yml --runs 5 --num_layers $LAYERS 2>&1 | tee -a logs/train_${DATASET}_nlayers.log
done
done


### debug
# CUDA_VISIBLE_DEVICES="6" CUDA_LAUNCH_BLOCKING=1 python -u baseline.py --config config/weibo.yml --device cpu --runs 1 


cd -