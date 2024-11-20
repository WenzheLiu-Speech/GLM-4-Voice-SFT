# -*- coding:utf-8 -*-
# CopyRight 刘聪NLP, 2023
# CopyRight wenzheliu, 2024

import torch
import random
import numpy as np
from transformers import set_seed
import json
import os
from torch.utils.data import Dataset


class GLMSFTDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, is_skip):
        end_token_id = tokenizer.convert_tokens_to_ids('<|user|>')
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip()) 
                skip_flag = False
                input_ids = sample['input_ids']

                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    skip_flag = True

                labels = sample['labels']

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids('<|user|>') # tokenizer.pad_token_id

    def __call__(self, batch):
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = max(lengths)

        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long)}

