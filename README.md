

## Introduction:
This is the code for the ICLR 2024 paper of **ConsisGAD**: [Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision.](https://openreview.net/forum?id=elMKXvhhQ9). 


## Data Preparation

> If you use Amazon, Yelp or TFinance datasets, your may not need to do this step. 

This repository uses DGL to load graphs. The data needs to be converted to a format acceptable to the DGL

For example, when your data is in `CSV` format, see the guidance in [https://docs.dgl.ai/en/0.8.x/guide/data-loadcsv.html](https://docs.dgl.ai/en/0.8.x/guide/data-loadcsv.html). 

Then, write your dataloader in `./modules/data_loader.py`. 


## Train

The configuration file should be put in the `./config` directory. 

Then, run

```bash
chmod +x scripts/run.sh
cd scripts && ./run.sh
```

The results should be stored at the `./results` directory. The trained model should be stored at the `model-weights` directory. 

## Evaluation on your test set

For example, when you want to evaluate the trained model on T2 test set, you may need to write your own evaluate function in baseline.py. Please refer to the `run_model` function in `baseline.py`. 




