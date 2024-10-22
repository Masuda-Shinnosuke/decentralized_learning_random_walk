import argparse
import random
import numpy as np
import torch

def parse_argument():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_set",default = "cifar-10")
    parser.add_argument("-s","--seed",default = "2024")
    parser.add_argument("-n","--num_nodes",default="10")
    parser.add_argument("-bs","--batch_size",default="256")
    parser.add_argument("-hk","--hetero_k",default="10")
    parser.add_argument("-gh","--graph_type",default="ring")
    parser.add_argument("-mo","--model",default="cnn")
    parser.add_argument("-l","--learning_rate",default="0.01")
    parser.add_argument("-t","--iteration",default="100")
    args = parser.parse_args()


    return args

def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True