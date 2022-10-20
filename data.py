import os
import json
import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = '<PAD>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
MASK_TOKEN = '<MASK>'

class Vocabulary(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size=None, tokenizer=None, mask_token=False):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
        :param vocab_file: string; path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first.
        :param max_size: int; The maximum size of the resulting Vocabulary.
        :param tokenizer: tokenizer used to split sentences
        :param mask_token:  whether to add MASK token
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab
        self._num = 0  # read number of words
        self._tokenizer = get_tokenizer("basic_english") if tokenizer==None else tokenizer

        for w in [PAD_TOKEN, SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf-8') as vocab_f: # New : add the utf8 encoding to prevent error
            cnt = 0
            for line in vocab_f:
                cnt += 1
                pieces = line.strip().split(" ")
                w = pieces[0]
                if w in self._word_to_id:
                    raise ValueError('Duplicated word in vocabulary file Line {} : {}'.format(cnt, w))
                    cnt -= 1
                    continue
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != None and self._count >= max_size:
                    print("max_size of vocab was specified as {}; we now have {} words. Stopping reading.".format(max_size, self._count))
                    break
            self._num = cnt
        print("Finished constructing vocabulary of {} total words. Last word added: {}".format(self._count, self._id_to_word[self._count-1]))
        
        if mask_token:
            self._word_to_id[MASK_TOKEN] = self._count
            self._id_to_word[self._count] = MASK_TOKEN
            self._count += 1
            print("Add Mask token {} with id={} (vocab_size={})".format(MASK_TOKEN, self._word_to_id[MASK_TOKEN], self._count))

    def _token2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def _id2token(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: {}'.format(word_id))
        return self._id_to_word[word_id]

    def detokenize(self, ids):
        tokens = [self._id2token(x) for x in ids]
        return tokens

    def tokenize(self, tokens):
        if isinstance(tokens, str):
            tokens = self._tokenizer(tokens)
        ids = [self._token2id(x) for x in tokens]
        return ids

    def get_vocab_dict(self):
        return self._word_to_id

    def __len__(self):
        return self._count



class TextDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.dataset = []

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def _load(self, path):
        if not os.path.exists(path):
            raise KeyError('The path {} does not exist'.format(path))
        data = []
        if path.endswith('json'):
            with open(path) as fin:
                for line in fin:
                    data.append(json.loads(line))
        else:
            with open(path) as fin:
                for line in fin:
                    data.append(line.strip())
        return data

class CLSDataset(TextDataset):
    def __init__(self, mode, path, vocab):
        super().__init__(path)
        self.mode = mode
        self.vocab = vocab
        self.raw_dataset = self.load(path)
        self.dataset = self.preprocess(self.raw_dataset)

    def preprocess(self, dataset):
        tokenized_dataset = []
        for x, y in dataset:
            parts = self.vocab.tokenize(x)
            tokenized_dataset.append((parts, y))
        return tokenized_dataset

    def _get_raw_pair(self, idx):
        return self.raw_dataset[idx]

    def load(self, path):
        if self.mode == 'train':
            sentences_pos = self._load(f"{path}/training_pos.txt")
            sentences_neg = self._load(f"{path}/training_neg.txt")
        elif self.mode == 'test':
            sentences_pos = self._load(f"{path}/test_pos_public.txt")
            sentences_neg = self._load(f"{path}/test_neg_public.txt")

        sentences = sentences_pos + sentences_neg
        labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]
        return [(x, y) for x, y in zip(sentences, labels)]

class LMDataset(TextDataset):
    def __init__(self, mode, path, vocab, maxlen=None):
        super().__init__(path)
        self.mode = mode
        self.vocab = vocab
        self.maxlen = maxlen
        self.raw_dataset = self.load(path)
        self.dataset = self.preprocess(self.raw_dataset)

    def preprocess(self, dataset):
        tokenized_dataset = []
        BOS, EOS = self.vocab._token2id(SENTENCE_START), self.vocab._token2id(SENTENCE_END)
        for x in dataset:
            parts = self.vocab.tokenize(x)
            if self.maxlen:
                parts = parts[:self.maxlen-2]
            sent = [BOS] + parts + [EOS]
            tokenized_dataset.append((sent, sent))
        return tokenized_dataset

    def _get_raw_pair(self, idx):
        return self.raw_dataset[idx]

    def load(self, path):
        if self.mode == 'train':
            sentences_pos = self._load(f"{path}/training_pos.txt")
            sentences_neg = self._load(f"{path}/training_neg.txt")
        elif self.mode == 'test':
            sentences_pos = self._load(f"{path}/test_pos_public.txt")
            sentences_neg = self._load(f"{path}/test_neg_public.txt")

        sentences = sentences_pos + sentences_neg
        return sentences

def collate_func(batch):
    x, y = zip(*batch)
    x = [torch.LongTensor(i) for i in x]
    batch_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    if isinstance(y[0], list):
        y = [torch.LongTensor(i) for i in y]
        batch_y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
    else:
        batch_y = torch.LongTensor(y)
    return batch_x, batch_y
    
