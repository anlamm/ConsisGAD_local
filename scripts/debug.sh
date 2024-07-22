

#!/bin/bash

cd ..

#### For testing / debugging
# CUDA_VISIBLE_DEVICES="5" python -u pretrain_merge.py --config config/amazon.yml --runs 5 --random_feature  --debug #### Use amazon's config
CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/amazon.yml --runs 5 --random_feature   --debug --drop_edges
CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/yelp.yml --runs 5 --random_feature   --debug --drop_edges
CUDA_VISIBLE_DEVICES="5" python -u baseline.py --config config/tfinance.yml --runs 5 --random_feature   --debug --drop_edges



cd -