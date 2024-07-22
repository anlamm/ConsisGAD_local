

## Introduction:
This is the code for the ICLR 2024 paper of **ConsisGAD**: [Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision.](https://openreview.net/forum?id=elMKXvhhQ9). 


## Data Preparation

This repository uses DGL to load graphs. The data needs to be converted to a format acceptable to the DGL

For example, when your data is in `CSV` format, see the guidance in [https://docs.dgl.ai/en/0.8.x/guide/data-loadcsv.html](https://docs.dgl.ai/en/0.8.x/guide/data-loadcsv.html). 

Then, write your dataloader in `./modules/data_loader.py`. 


## Train

The configuration file should be put in the `./config` directory. 

Then, run

```bash
chmod +x scripts/run_baseline.sh
cd scripts && ./run_baseline.sh
```

The results should be stored at the `./results` directory. 




