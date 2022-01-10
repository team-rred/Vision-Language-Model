import json
import os
from transformers import BertTokenizerFast
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

plt.hist(len_findings)
plt.show()



print('end')