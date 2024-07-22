

import os
import numpy as np
import pandas as pd

def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id

def merge_results(fileids=None, ext="xlsx", outname="", axis=0):    
    ##### axis=0: Merge results of different datasets
    ##### axis=1: Merge results of different methods

    root = "/home/yliumh/github/ConsisGAD/results"

    df_merged = None
    for fileid in fileids:
        filename = f"{fileid}.{ext}"
        if ext == "csv":
            df = pd.read_csv(f"{root}/{filename}", header=1)
        elif ext == "xlsx":
            xls = pd.ExcelFile(f"{root}/{filename}")
            df = pd.read_excel(xls, "Sheet1", header=1)
        else:
            raise NotImplementedError

        if df_merged is None:
            df_merged = df
        else:
            if axis==1 or axis==-1:
                df = df.iloc[:, 1:] #### 
            df_merged = pd.concat([df_merged, df], axis=axis)
    
    if ext == "csv":
        df_merged.to_csv(f"{root}/{outname}.{ext}", index=False)
    elif ext == "xlsx":
        df_merged.to_excel(f"{root}/{outname}.{ext}", index=False)
    else:
        raise NotImplementedError

    
if __name__ == "__main__":

    ###### Merge results of different datasets
    merge_results(fileids=[75,76,77], outname="baseline", axis=0)


    ###### Merge results of different methods on Amazon dataset
    merge_results(fileids=[3,7,13,8,18,21,24,27], outname="ablation-amazon", axis=1)

    ###### Merge results of different methods on Yelp dataset
    merge_results(fileids=[9,11,14,16,19,22,25,28], outname="ablation-yelp", axis=1)

    ###### Merge results of different methods on T-Finance dataset
    merge_results(fileids=[10,12,15,17,20,23,26,29], outname="ablation-tfinance", axis=1)



    ###### Merge results of different datasets
    merge_results(fileids=[33,34,35], outname="pretrain_merge_100", axis=0)

    ###### Merge results of different datasets
    merge_results(fileids=[78,79,80], outname="pretrain_merge_300", axis=0)