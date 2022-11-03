# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# 搭建模型
class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(dropout)
        self.bert = AutoModel.from_pretrained("xlm-roberta-base")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls = self.drop(outputs[1])
        out = F.sigmoid(self.fc(cls))
        return out








