from flogging import logger
import string
import numpy as np
import torch.nn as nn

DATAPATH="/home/danqingwang/Dataset/hw1/"


############# utility functions  #############

def load_data(file_name):
    '''
    @input:
     file_name: a string. should be either "training.txt" or "texting.txt"
    @return:
     a list of sentences
    '''
    with open(file_name, "r") as file:
        sentences = file.readlines()
    return sentences


def tokenize(sentence):
    # Convert a sentence into a list of words
    wordlist = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip().split(
        ' ')

    return [word.strip() for word in wordlist]


def loadDataset():
    logger.info("Creating a classifier agent:")

    with open(f"{DATAPATH}/vocab.txt") as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        vocab_dict = {item: i for i, item in enumerate(vocab_list)}

    logger.info("Loading and processing data ...")

    sentences_pos = load_data(f"{DATAPATH}/training_pos.txt")
    sentences_neg = load_data(f"{DATAPATH}/training_neg.txt")

    train_sentences = sentences_pos + sentences_neg
    train_sentences = [[vocab_dict[x] for x in tokenize(sent) if x in vocab_dict] for sent in train_sentences]
    train_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    sentences_pos = load_data(f"{DATAPATH}/test_pos_public.txt")
    sentences_neg = load_data(f"{DATAPATH}/test_neg_public.txt")
    test_sentences = sentences_pos + sentences_neg
    test_sentences = [[vocab_dict[x] for x in tokenize(sent) if x in vocab_dict] for sent in test_sentences]
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    return train_sentences, train_labels, test_sentences, test_labels, vocab_dict

def load_w2v(vocab_dict, embed_size):
    embed_weight = np.zeros((len(vocab_dict), embed_size))
    with open(f'{DATAPATH}/glove.6B.50d.txt') as fin:
        for line in fin:
            parts = line.split(' ')
            word = parts[0]
            if word in vocab_dict:
                embed = [float(x) for x in parts[1:]]
                assert len(embed) == embed_size
                embed_weight[vocab_dict[word],:] = embed
    return embed_weight


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def init_module(net):
    for module in net.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    

def init_param(net):
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            p.data.zero_()