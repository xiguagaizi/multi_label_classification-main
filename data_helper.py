# -*- coding: utf-8 -*-
# 数据编码化
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from data_preprocess import load_json
import pandas as pd


class MultiClsDataSet(Dataset):
    def __init__(self, data_path, max_len=128, label2idx_path="./data/label2idx.json"):
        # 加载json
        self.label2idx = load_json(label2idx_path)
        self.class_num = len(self.label2idx)
        # 载入预训练模型实例化一个tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        self.max_len = max_len
        self.input_ids, self.attention_mask, self.labels = self.encoder(data_path)

    def encoder(self, data_path):
        texts = []
        labels = []
        # 按行读取每条数据
        for index, item in pd.read_excel(data_path).iterrows():
            texts.append(item["Sentences"])
            labels.append(
                [item["anger"], item["disgust"], item["fear"], item["sadness"], item["surprise"], item["happiness"],
                 item["neutral"]])
        # with open(data_path, encoding="utf-8") as f:
        #     for line in f:
        #         line = json.loads(line)
        #         texts.append(line["text"])
        #         tmp_label = [0] * self.class_num
        #         for label in line["label"]:
        #             tmp_label[self.label2idx[label]] = 1
        #         labels.append(tmp_label)

        tokenizers = self.tokenizer(texts,
                                    # 给序列补全到一定长度，True or ‘longest’: 是补全到batch中的最长长度，max_length’:补到给定max - length或没给定时，补到模型能接受的最长长度。
                                    padding=True,
                                    # 截断操作，true or ‘longest_first’：给定max_length时，按照max_length截断，没给定max_lehgth时，到，模型接受的最长长度后截断，适用于所有序列（单或双）。only_first’：这个只针对第一个序列。only_second’：只针对第二个序列。
                                    truncation=True,
                                    max_length=self.max_len,
                                    # 返回数据的类型，可选tf’, ‘pt’ or ‘np’ ，分别表示tf.constant, torch.Tensor或np.ndarray
                                    return_tensors="pt",
                                    is_split_into_words=False)
        # {返回字典
        #     input_ids: list[int],
        #     token_type_ids: list[int] if return_token_type_ids is True(default)
        #     attention_mask: list[int] if return_attention_mask is True(default)
        #     overflowing_tokens: list[int] if the tokenizer is a slow tokenize, else a List[List[int]] if a
        #     special_tokens_mask: list[int] if ``add_special_tokens``
        # }
        # 取出input_ids
        input_ids = tokenizers["input_ids"]
        attention_mask = tokenizers["attention_mask"]

        return input_ids, attention_mask, \
               torch.tensor(labels, dtype=torch.float)

    # 获取标签长度
    def __len__(self):
        return len(self.labels)

    # 按照序号获取返回字典内容
    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item], self.labels[item]


if __name__ == '__main__':
    dataset = MultiClsDataSet(data_path="./data/train.xlsx")
    # 输入特征
    print(dataset.input_ids)
    # 目标特征
    print(dataset.attention_mask)

    print(dataset.labels)
