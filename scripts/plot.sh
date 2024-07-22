#!/bin/bash



cd ..

[ -d logs ] || mkdir logs


#### Plot using trained models
CUDA_VISIBLE_DEVICES="7" python -u plot.py --config config/amazon.yml --runs 1  --load_model 2>&1 | tee logs/plot.log
CUDA_VISIBLE_DEVICES="7" python -u plot.py --config config/yelp.yml --runs 1  --load_model 2>&1 | tee -a logs/plot.log
CUDA_VISIBLE_DEVICES="7" python -u plot.py --config config/tfinance.yml --runs 1  --load_model 2>&1 | tee -a logs/plot.log



#### Plot using random initialized models
CUDA_VISIBLE_DEVICES="7" python -u plot.py --config config/amazon.yml --runs 1 2>&1 | tee -a logs/plot.log
CUDA_VISIBLE_DEVICES="7" python -u plot.py --config config/yelp.yml --runs 1 2>&1 | tee -a logs/plot.log
CUDA_VISIBLE_DEVICES="7" python -u plot.py --config config/tfinance.yml --runs 1 2>&1 | tee -a logs/plot.log



cd -