# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from data_helper import MultiClsDataSet
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,hamming_loss


train_path = "./data/train.xlsx"
dev_path = "./data/dev.xlsx"
test_path = "./data/test.xlsx"
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
class_num = len(label2idx)
device = "cuda" if torch.cuda.is_available() else "cpu"
#模型超参数
lr = 2e-5 #回归参数
batch_size = 32 #每批数据量的大小
max_len = 128
hidden_size = 768
epochs = 31 #迭代次数
# 一个excel中包括200个样本（数据行）的数据，选择batch_size=5, epoch=1000,
# 则batch=40个，每个batch有5个样本，一次epoch将进行40个batch或40次模型参数更新，1000个epoch，
# 模型将传递整个数据集1000次，在整个训练过程中，总共有40000次batch.

# 获取数据
train_dataset = MultiClsDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
dev_dataset = MultiClsDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)

# 传入的数据集，每个batch有多少个样本，在每个epoch开始的时候，是否对数据进行重新排序
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

#获取Accuracy准确率
def get_acc_score(y_true_tensor, y_pred_tensor):
    y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    y_true_tensor = y_true_tensor.cpu().numpy()
    return accuracy_score(y_true_tensor, y_pred_tensor)

# 模型训练
def train():
    # 加载模型
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.train()
    model.to(device)

    # AdamW优化器优化，加快训练速度
    optimizer = AdamW(model.parameters(), lr=lr)
    # 计算衰减
    criterion = nn.BCELoss()
    ExpLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=0,
                                              verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                              min_lr=0, eps=1e-08)

    # 验证集最佳准确率
    dev_best_f1 = -1.
    # 迭代n次
    for epoch in range(1, epochs):
        model.train()

        for i, batch in enumerate(train_dataloader):
            # 梯度清零
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            # 获取标签文本
            labels = batch[-1]
            # 用迭代模型获取特征向量
            logits = model(*batch[:2])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # 每一百步输出
            if i % 100 == 0:
                acc_score = get_acc_score(labels, logits)
                # print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))

        # 验证集合
        dev_loss, dev_acc,dev_pre,dev_rec,dev_f1_micro,dev_f1_macro = dev(model, dev_dataloader, criterion)
        ExpLR.step(dev_f1_macro)  # 每个epoch衰减一次学习率
        # 写入每次迭代分数
        with open("./model/output.log","a+") as f :
            f.write("Dev epoch:{} acc:{:.3f} pre:{:.3f} rec:{:.3f} f1_micro:{:.3f}  f1_macro:{:.3f} loss:{:.3f} \n".format(epoch, dev_acc,dev_pre,dev_rec,dev_f1_micro,dev_f1_macro, dev_loss))
        print(optimizer.param_groups[0]["lr"])
        print("Dev epoch:{} acc:{:.3f} pre:{:.3f} rec:{:.3f} f1_micro:{:.3f}  f1_macro:{:.3f} loss:{:.3f} \n".format(epoch, dev_acc,dev_pre,dev_rec,dev_f1_micro,dev_f1_macro, dev_loss))
        #验证模型性能，若验证集的评价为当前对高，则保存当前训练模型
        if dev_f1_macro > dev_best_f1:
            dev_best_f1 = dev_f1_macro
            torch.save(model.state_dict(), save_model_path)



# 模型验证
def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids,  attention_mask, labels = [d.to(device) for d in batch]
            logits = model(input_ids,  attention_mask)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels_tensor = torch.cat(true_labels, dim=0)
    pred_labels_tensor = torch.cat(pred_labels, dim=0)
    y_pred = (pred_labels_tensor.cpu() > 0.5).int().numpy()
    y_true = true_labels_tensor.cpu().numpy()
    acc_score=accuracy_score(y_true,y_pred)
    pre_score=precision_score(y_true,y_pred,average="micro")
    rec_score=recall_score(y_true,y_pred,average="micro")
    f1_micro_score = f1_score(y_true, y_pred, average="micro")
    f1_macro_score=f1_score(y_true,y_pred,average="macro")

    return np.mean(all_loss), acc_score,pre_score,rec_score,f1_micro_score,f1_macro_score

#模型测试，多余，重新写到predict上去
def test(model_path, test_data_path):
    test_dataset = MultiClsDataSet(test_data_path, max_len=max_len, label2idx_path=label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, attention_mask, labels = [d.to(device) for d in batch]
            logits = model(input_ids,  attention_mask)
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return acc_score


if __name__ == '__main__':
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    train()
