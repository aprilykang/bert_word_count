import argparse
import random
from re import X

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel
from Hbert_trainer import BertTrainer as Trainer
from Hbert_dataset import TextClassificationDataset, TextClassificationCollator
from Hbert_utils import read_text
#from Hbert_model import Hbert

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn')
    p.add_argument('--train_fn', default='/home/jwp/home/jwp/project/train.csv')
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--pretrained_model_name', type=str, default="bert-base-uncased")
    p.add_argument('--gpu_id', type=int, default=1)
    p.add_argument('--verbose', type=int, default=2) 
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--n_layer', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-6)
    
    p.add_argument('--warmup_ratio', type=float, default=5e-2)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--max_length', type=int, default=512)

    config = p.parse_args()

    return config

def get_loaders(fn, tokenizer, valid_ratio=.2):
    labels, texts = read_text(fn)

    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    
    # Generate label to index map.
    index_to_label = {
        0: "ele",
        1: "in",
        2: "adv"
    }

    return train_loader, valid_loader, index_to_label


def get_optimizer(model, config):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
        eps=config.adam_epsilon
    )

    return optimizer


def main(config):
    # Get pretrained tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders(
        config.train_fn, tokenizer,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

     # Get pretrained model with specified softmax layer.
    model_loader = BertForSequenceClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )
    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
