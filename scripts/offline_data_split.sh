#!/bin/bash


cd ..

[ -d logs ] || mkdir logs

python -u modules/data_loader.py 2>&1 | tee -a logs/offline_split.log




cd -