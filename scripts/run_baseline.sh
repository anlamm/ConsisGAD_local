#!/bin/bash


CUDA_VISIBLE_DEVICES="0" python -u baseline.py --config config/[YOUR DATA].yml --runs 5  2>&1 | tee logs/train_[YOUR DATA].log
