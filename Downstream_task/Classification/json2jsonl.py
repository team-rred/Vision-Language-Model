import pandas as pd
import sys
from scipy.sparse.construct import random
import os

# path_dev = "/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr/dev.json"
# path_train = "/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr/train.json"
# PATH = sys.argv[1]
# IS_ERROR = bool(sys.argv[2])
PATH = '/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr'
FILE = 'train_error_12_29.json'
IS_ERROR = False
# data_dev = pd.read_json(path_dev).dropna()
data = pd.read_json(os.path.join(PATH,FILE)).dropna()

# data_dev['label']=0
# data['label'] = 1 if IS_ERROR else 0

bool_idx = data['error_label']!=0
data = data.loc[bool_idx] #85057개
data['label'] = data['error_label']
# data['label'] = 1


from sklearn.model_selection import train_test_split
train, dev = train_test_split(data, test_size=0.2, random_state=1125)



# with open("/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr/dev.jsonl", "w") as file:
#     file.write(data_dev.to_json(orient='records', lines=True))
with open(os.path.join(PATH,'train_12_29_error.jsonl'), "w") as file:
    file.write(train.to_json(orient='records', lines=True))
with open(os.path.join(PATH,'dev_12_29_error.jsonl'), "w") as file:
    file.write(dev.to_json(orient='records', lines=True))


print('jsonl is saved at ', PATH+'l')

PATH = '/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr'
FILE = 'train.json'
data = pd.read_json(os.path.join(PATH,FILE)).dropna()

data = data.loc[bool_idx] #85057개
data['label'] = 0


train, dev = train_test_split(data, test_size=0.2, random_state=1125)

with open(os.path.join(PATH,'train_12_29_normal.jsonl'), "w") as file:
    file.write(train.to_json(orient='records', lines=True))
with open(os.path.join(PATH,'dev_12_29_normal.jsonl'), "w") as file:
    file.write(dev.to_json(orient='records', lines=True))