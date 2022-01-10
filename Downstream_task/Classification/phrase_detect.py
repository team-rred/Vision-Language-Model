import json
import os
from transformers import BertTokenizerFast
from tqdm import tqdm
import numpy as np



test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.  thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession ession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.  thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child  """.split()

bigram = [(test_sentence[i], test_sentence[i + 1]) for i in range(len(test_sentence) - 1)]
bigram = [(test_sentence[i], test_sentence[i + 1]) for i in range(len(test_sentence) - 1)]
trigram = [(test_sentence[i], test_sentence[i + 1], test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]


import math
import collections
import torch

def ngrams_iterator(token_list, ngrams):
    """Return an iterator that yields the given tokens and their ngrams.

    Arguments:
        token_list: A list of tokens
        ngrams: the number of ngrams.

    Examples:
        >>> token_list = ['here', 'we', 'are']
        >>> list(ngrams_iterator(token_list, 2))
        >>> ['here', 'here we', 'we', 'we are', 'are']
    """

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    # for x in token_list:
    #     yield x
    for n in range(ngrams, ngrams + 1):
        for x in _get_ngrams(n):
            yield ' '.join(x)

def _compute_ngram_counter(tokens, max_n):
    """ Create a Counter with a count of unique n-grams in the tokens list

    Arguments:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted

    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count

    Examples:
        >>> from torchtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['me', 'me', 'you']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1,
             ('me', 'me', 'you'): 1})
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(tuple(x.split(' '))
                                         for x in ngrams_iterator(tokens, max_n))

    return ngrams_counter


bi_counter =  _compute_ngram_counter(test_sentence, 2)
tri_counter =  _compute_ngram_counter(test_sentence, 3)

ngram_counter_list = [bi_counter, tri_counter]
top_n = [2, 1]
def get_ngram_vocab(ngram_counter_list:list, top_n:list):
    ngram_counter.most_common()


tokenizer = (
    BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True).tokenize
)

data_path = '/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr'
Train_dset0_name = 'train_12_29_normal.jsonl'

data_path = os.path.join(data_path, Train_dset0_name)

data = [json.loads(l) for l in open(data_path)]

findings = [tokenizer(x['findings']) for x in tqdm(data)]
impression = [tokenizer(x['impression']) for x in tqdm(data)]

len_findings = [len(x) for x in findings]
len_impression = [len(x) for x in impression]




print('end')