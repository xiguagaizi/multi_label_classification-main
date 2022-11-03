# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from transformers import AutoTokenizer

hidden_size = 768
class_num=7
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
idx2label = {idx: label for label, idx in label2idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
max_len = 128

model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
model.load_state_dict(torch.load(save_model_path,map_location="cpu"))
model.to(device)
model.eval()


def predict(texts):
    outputs = tokenizer(texts, return_tensors="pt", max_length=max_len,
                        padding=True, truncation=True)
    logits = model(outputs["input_ids"].to(device),
                   outputs["attention_mask"].to(device)
                  )
    logits = logits.cpu().tolist()

    result = []

    for sample in logits:
        for idx, logit in enumerate(sample):
            if logit > 0.5:
                result.append(1)
            else:
                result.append(0)

    return result


if __name__ == '__main__':
    test_path='./data/test.xlsx'
    df =pd.read_excel(test_path)
    texts =np.array(df["Sentences"]).tolist()
    result=[]
    for text in texts:

        temp= predict(text)
        result.append(temp)
    df_predict=pd.DataFrame(result,columns=["anger","disgust","fear","sadness",	"surprise",	"happiness","neutral"])
    df_result=pd.concat([df_predict,df],axis=1)
    df_result.to_excel("./model/out.xlsx",index=False)
    print(result)


