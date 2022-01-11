import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('self.max_seq_len:', self.max_seq_len)
        # print('args.num_image_embeds:', self.args.num_image_embeds)
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data[index]["text"])[
                : (self.max_seq_len - 1)
            ] + self.text_start_token
        )
        segment = torch.zeros(len(sentence))
        # print('sentence:', sentence)
        # print('len_seq:', len(sentence))
        # print('**************************************')

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            if self.data[index]["label"] == '':
                self.data[index]["label"] = "'Others'"
            else:
                pass  
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"].split(', ')]
            ] = 1
        else:
            input("이거는 multilabel 학습 아니니 다시 돌리세요~")
            pass

        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            if self.data[index]["img"]:
                image = Image.open(
                    os.path.join(self.data_dir, self.data[index]["img"]))
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment, image, label


class JsonlDatasetSNUH(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args, report_img_pair_info, data_path_1):
        self.data = [json.loads(l) for l in open(data_path)]

        #### TEMP error file, 나중에는 data 불러온 다음에 tool 이용해서 에러 추가하도록 해야함.
        self.data_error = [json.loads(l) for l in open(data_path_1)]
        self.data = self.temp_error_sampler()
        #
        
        self.data_dir = os.path.dirname(data_path)
        self.data_dir_img = os.path.join(self.data_dir, 'mimic-nlp-jpg')
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len_findings = args.max_seq_len_findings
        self.max_seq_len_impression = args.max_seq_len_impression
        if args.model == "mmbt":
            self.max_seq_len_findings -= int(args.num_image_embeds/2)
            self.max_seq_len_impression -= int(args.num_image_embeds/2)

        self.transforms = transforms

        self.r2i = report_img_pair_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('self.max_seq_len:', self.max_seq_len)
        # print('args.num_image_embeds:', self.args.num_image_embeds)
        sentence_findings = (
            self.text_start_token
            + self.tokenizer(self.data[index]["findings"])[
                : (self.max_seq_len_findings-1)
            ] + self.text_start_token
        )
        segment_findings = torch.zeros(len(sentence_findings))
        # print('sentence:', sentence)
        # print('len_seq:', len(sentence))
        # print('**************************************')

        sentence_findings = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence_findings
            ]
        )

        sentence_impression = (
            self.text_start_token
            + self.tokenizer(self.data[index]["impression"])[
                : (self.max_seq_len_impression - 1)
            ] + self.text_start_token
        )
        segment_impression = torch.zeros(len(sentence_impression))
        # print('sentence:', sentence)
        # print('len_seq:', len(sentence))
        # print('**************************************')

        sentence_impression = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence_impression
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            if self.data[index]["label"] == '':
                self.data[index]["label"] = "'Others'"
            else:
                pass  
            label[
                [self.args.labels.index(tgt) for tgt in str(self.data[index]["label"]).split(', ')]
            ] = 1
        elif self.args.task_type == "classification":
            label = self.args.labels.index(str(self.data[index]['label']))
            # label = torch.zeros(self.n_classes)
            # label[
            #     self.args.labels.index(str(self.data[index]['label']))
            # ] = 1
        elif self.args.task_type == "binary":
            if self.data[index]['label'] != 0:
                self.data[index]['label'] = 1
            label = self.args.labels.index(str(self.data[index]['label']))
        else:
            input("이거는 multilabel 학습 아니니 다시 돌리세요~")
            pass

        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            ########## mimic-cxr-nlp_210904.csv 불러와서 참조해야함

            ssid = f"{self.data[index]['subject_id']}_{self.data[index]['study_id']}"
            image_file_name = (ssid + '_' + self.r2i[self.r2i.ssid==ssid].frontal.values[0]+'.jpg') if ssid in self.r2i.ssid.values and self.r2i[self.r2i.ssid==ssid].frontal.values[0]!='NAN' else None

            if image_file_name is None:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            else:
                image = Image.open(
                    os.path.join(self.data_dir_img,image_file_name))
                
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment_findings = segment_findings[1:]
            sentence_findings = sentence_findings[1:]
            # The first segment (0) is of images.
            segment_findings += 1

            # The first SEP is part of findings Token.
            segment_impression = segment_impression[1:]
            sentence_impression = sentence_impression[1:]
            # The first segment (0) is of findings.
            segment_impression += 2

            sentence = torch.cat((sentence_findings, sentence_impression),0)
            segment = torch.cat((segment_findings, segment_impression),0)

        return sentence, segment, image, label


    def temp_error_sampler(self):
        print("Generating Error...............")
        
        import random
        data_with_error = []
        for idx in range(len(self.data)):
            add_error = bool(random.randint(0,1))
            if add_error:
                data_with_error.append(self.data_error[idx])
            else: 
                data_with_error.append(self.data[idx])

        
        return data_with_error

    def temp_get_labels_and_frequencies(self):
        from collections import Counter

        label_freqs = Counter()
        if self.args.task_type == "classification":
            data_labels = [str(line["label"]) for line in self.data]
        elif self.args.task_type == "binary":
            data_labels = ["1" if line["label"]!=0 else "0" for line in self.data]
            
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