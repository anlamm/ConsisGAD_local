import numpy as np
import torch
import torch.nn.functional as F
import dgl
import os
import math
import pickle
from sklearn.metrics import f1_score


def to_np(x):
    return x.cpu().detach().numpy()


def store_model(my_model, args):
    os.makedirs(f"{args['save_path']}", exist_ok=True)
    file_path = os.path.join(f"{args['save_path']}",
                             f"{args['data-set']}-{args['seed']}.pth")
    torch.save(my_model.state_dict(), file_path)
