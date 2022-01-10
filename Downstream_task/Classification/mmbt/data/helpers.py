import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from data.dataset import JsonlDataset, JsonlDatasetSNUH
from data.vocab import Vocab

import pandas as pd


def get_transforms(args):
    if 1: ##args.openi:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [str(json.loads(line)["label"]) for line in open(path)]
    if type(data_labels) == list:
        for label_row in data_labels:
            if label_row == '':
                label_row = ["'Others'"]
            else:
                label_row = label_row.split(', ')

            label_freqs.update(label_row)
    else:
        pass
    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    elif args.task_type =='classification':
        # Mulitclass case
        tgt_tensor = torch.tensor([row[3] for row in batch]).long()
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert"]
        else str.split
    )

    transforms = get_transforms(args)

#     args.labels, args.label_freqs = get_labels_and_frequencies(
#         os.path.join(args.data_path, args.Train_dset_name)
#     )

# ###############################TEMP
    args.labels = [0,1,2,3]
#     args.label_freqs = Counter({'0':45772,'1':45772})
#################################

    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz

########## report img pair 불러와서 참조
    report_img_pair_info_path = '/home/workspace/Multi-modality-Self-supervision/data/mimic-cxr/mimim-cxr-nlp_210904.csv'
    report_img_pair_info = pd.read_csv(report_img_pair_info_path)
############

    train = JsonlDatasetSNUH(
        os.path.join(args.data_path, args.Train_dset0_name),################TEMP
        tokenizer,
        transforms,
        vocab,
        args,
        report_img_pair_info,
        os.path.join(args.data_path, args.Train_dset1_name),################TEMP
    )

    args.train_data_len = len(train)

    dev = JsonlDatasetSNUH(
        os.path.join(args.data_path, args.Valid_dset0_name),################TEMP
        tokenizer,
        transforms,
        vocab,
        args,
        report_img_pair_info,
        os.path.join(args.data_path, args.Valid_dset1_name),################TEMP
    )

    args.labels, args.label_freqs = train.temp_get_labels_and_frequencies()
    args.n_classes = len(args.labels)

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    return train_loader, val_loader  # , test

