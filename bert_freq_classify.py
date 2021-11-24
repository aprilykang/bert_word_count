import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
import json


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_text():
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip()]

    return lines


def main(config):
    with open('/home/jwp/home/jwp/project/HBERT/Hbert/vocab_v2.json') as f:
      vocab_freq = json.load(f)

    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = read_text()

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = BertForSequenceClassification
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            #print(mini_batch['attention_mask'])
            

            tokenized_text=tokenizer.tokenize(str(lines[idx:idx + config.batch_size]))
            
            info=vocab_freq
            temp=[]
            #temp.append(1)
            temp.append(1)
            temp = [1/int(info[i]) for i in tokenized_text]
           # print(temp)
            while(len(temp) < 511):
                temp.append(0)
            #mini_batch['attention_mask'][0][1:-1]=torch.FloatTensor(temp[0:510])
            #print(torch.tensor(temp[0:510]))
            

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            #mask=torch.tensor(temp[0:512])
            #mask=mask.unsqueeze(0)
            #print(mask)
            mask = mask.to(device)

            # Take feed-forward
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1)

            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)
        label2num = {
            "ele": 0,
            "in": 1,
            "adv": 2,
        }

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)

        print(label2num[index_to_label[int(indice[0][0])]])

        f = open("kaggle.txt", 'w')

        for i in range(len(lines)):
            sys.stdout.write('%s\t%d\n' % (
               lines[i],
               (label2num[index_to_label[int(indice[i][0])]])
            ))
            f.write('%d\n' % (
               (label2num[index_to_label[int(indice[i][0])]])
            ))
        f.close()

if __name__ == '__main__':
    config = define_argparser()
    main(config)
