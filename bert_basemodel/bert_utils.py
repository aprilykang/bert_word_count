import pandas as pd
import re
#import emoji
#from soynlp.normalizer import repeat_normalize
import torch
import random



def read_text(fn):

    def preprocess_dataframe(df):
        
        
        label2index = {
            "ele": 0,
            "in": 1,
            "adv": 2
        }
        df['label'] = df['label'].replace(label2index)
        return df

    with open(fn, 'r') as f:
        df = pd.read_csv(fn)
        df=preprocess_dataframe(df)
        labels = df.label.to_list()
        texts = df.text.to_list()

        print(labels)
        print(texts)
    return labels, texts


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
