import json
import os
from transformers import BertTokenizerFast
from tqdm import tqdm
import numpy as np
import itertools


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
import torch


def get_ngram_vocab(report_data, ngram_list:list,top_n_list:list):
    """n-gram vocab 생성

    Args:
        report_data ([type]): [description] 판독문 데이터
        ngram_list (list): [description] [2,3] 이면 bigram, trigram
        top_n_list (list): [description]

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """
    import collections

    ngram_vocab = []

    corpus_tokens = [(x['findings']+' '+x['impression']).split() for x in report_data]
    corpus_tokens = list(itertools.chain(*corpus_tokens))

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

        assert max_n > 0
        ngrams_counter = collections.Counter(tuple(x.split(' '))
                                            for x in tqdm(ngrams_iterator(tokens, max_n)))

        return ngrams_counter

    ngram_counter_list = [_compute_ngram_counter(corpus_tokens, n) for n in ngram_list]

    for ngram_counter, n in zip(ngram_counter_list, top_n_list):
        for ngram in ngram_counter.most_common(n):
            ngram_vocab.append(' '.join(ngram[0]))
            
    return ngram_vocab

# ngram_vocab = get_ngram_vocab(ngram_list=[2,3], top_n_list=[10,10])

data_path = 'data/mimic-cxr'
Train_dset0_name = 'train_12_29_normal.jsonl'
data_path = os.path.join(data_path, Train_dset0_name)
data = [json.loads(l) for l in open(data_path)]

ngram_vocab = get_ngram_vocab(report_data=data, ngram_list=[2,3], top_n_list=[10,10])

"""
쓸데없는게 많이 나온다. TF-IDF로 할까?
00:'There is'
01:'of the'
02:'in the'
03:'is no'
04:'pleural effusion'
05:'the right'
06:'No acute'
07:'effusion or'
08:'or pneumothorax.'
09:'contours are'
10:'There is no'
11:'pleural effusion or'
12:'No acute cardiopulmonary'
13:'and hilar contours'
14:'acute cardiopulmonary process.'
15:'effusion or pneumothorax.'
16:'The lungs are'
17:'hilar contours are'
18:'of the chest'
19:'or pneumothorax is'
"""


print('end')