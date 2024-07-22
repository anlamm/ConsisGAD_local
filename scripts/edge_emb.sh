#!/bin/bash

cd ..

# python -u edge_emb.py --config config/yelp.yml --runs 1 --mode train
# python -u edge_emb.py --config config/amazon.yml --runs 1 --mode train
# python -u edge_emb.py --config config/tfinance.yml --runs 1 --mode train


python -u edge_emb.py --config config/yelp.yml --runs 1 --mode test
python -u edge_emb.py --config config/amazon.yml --runs 1 --mode test
python -u edge_emb.py --config config/tfinance.yml --runs 1 --mode test


cd -