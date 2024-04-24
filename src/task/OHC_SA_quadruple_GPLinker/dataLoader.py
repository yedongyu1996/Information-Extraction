import json
import os

try:
    os.environ['project_root']
except KeyError:
    os.environ["project_root"] = os.path.abspath("../../../")
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import torch
from config import Config


def padding(l, length, t):
    """
    用（0,0）补全l
    """
    to_list = [list(i) for i in l]
    array = []
    for i in to_list:
        temp = i
        if len(i) < length:
            for j in range(length - len(i)):
                temp.append((0, 0))
        else:
            if len(i) > 1:
                if len(i) != length:
                    print(length)
                    print(len(i))
                    print(i)
                    print(t)
                # assert len(i) == lenth
        array.append(temp)
    return np.array(array)


class MyDataset(Dataset):
    def __init__(self, config, fn):
        self.config = config

        with open(os.environ["project_root"] + "/" + self.config.map_rel, "r",
                  encoding="utf-8") as f:
            self.label2id = json.load(f)[0]

        self.tokenizer = BertTokenizer.from_pretrained(
            os.environ["project_root"] + "/" + self.config.bert_path, do_lower_case=False)
        self.data = json.load(open(os.environ["project_root"] + "/" + fn, "r", encoding="utf-8"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_data = self.data[idx]
        text = ''.join(json_data["words"])
        maxlen = 510
        if len(text) > maxlen:
            text = text[:maxlen]
        token = ['[CLS]'] + list(text) + ['[SEP]']
        token_len = len(token)

        token2id = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * token_len

        attention_mask = np.array(mask)
        input_ids = np.array(token2id)

        # 实体
        entity_list = [[], []]  # 分别记录头尾

        # 类别
        head_category_list = [[] for _ in range(self.config.num_category)]
        tail_category_list = [[] for _ in range(self.config.num_category)]

        # 情感
        head_sentiment_list = [[] for _ in range(self.config.senti_polarity)]
        tail_sentiment_list = [[] for _ in range(self.config.senti_polarity)]

        quadruple_list = []

        for aspect, opinion in zip(json_data["aspects"], json_data["opinions"]):
            entity_list[0].append([aspect["from"] + 1, aspect["to"]])  # +1是因为前面加了CLS
            entity_list[1].append([opinion["from"] + 1, opinion["to"]])

            category = self.label2id[aspect['category']]
            head_category_list[category].append([aspect["from"] + 1, opinion["from"] + 1])
            tail_category_list[category].append([aspect["to"], opinion["to"]])

            sentiment = self.label2id[aspect['sentiment']]
            head_sentiment_list[sentiment].append([aspect["from"] + 1, opinion["from"] + 1])
            tail_sentiment_list[sentiment].append([aspect["to"], opinion["to"]])

            quadruple_list.append(
                [aspect["from"] + 1, aspect["to"], aspect["polarity"], aspect["category"], opinion["from"] + 1,
                 opinion["to"]])

        for label in entity_list + head_category_list + tail_category_list + head_sentiment_list + tail_sentiment_list:
            if not label:
                label.append([0, 0])

        en1 = len(entity_list[0])
        en2 = len(entity_list[1])
        if en1 < en2:
            for i in range(en2 - en1):
                entity_list[0].append([0, 0])
        else:
            for i in range(en1 - en2):
                entity_list[1].append([0, 0])
        entity_list = np.array(entity_list)  # 两个表，分别是所有头实体与尾实体的位置，两者元素之间并没有一一对应的一致关系
        entity_list_length = entity_list.shape[1]

        head_category_length = [len(i) for i in head_category_list]
        max_hcl = max(head_category_length)
        head_category_list = padding(head_category_list, max_hcl, text)

        tail_category_length = [len(i) for i in tail_category_list]
        max_tcl = max(tail_category_length)
        tail_category_list = padding(tail_category_list, max_tcl, text)

        head_sentiment_length = [len(i) for i in head_sentiment_list]
        max_hsl = max(head_sentiment_length)
        head_sentiment_list = padding(head_sentiment_list, max_hsl, text)

        tail_sentiment_length = [len(i) for i in tail_sentiment_list]
        max_tsl = max(tail_sentiment_length)
        tail_sentiment_list = padding(tail_sentiment_list, max_tsl, text)

        return (text, token, quadruple_list, input_ids, attention_mask, token_len, entity_list_length,
                entity_list, head_category_list, tail_category_list, max_hcl, max_tcl, head_sentiment_list,
                tail_sentiment_list, max_hsl, max_tsl)


def collate_fn(batch):
    (text, token, spo_list, input_ids, attention_mask, token_len, entity_length, entity_list, head_cate_list, tail_cate_list,
     head_cate_len, tail_cate_len, head_senti_list, tail_senti_list, head_senti_len, tail_senti_len) = zip(*batch)

    cur_batch = len(batch)
    max_text_length = max(token_len)
    max_entity_length = max(entity_length)
    max_head_cate_len = max(head_cate_len)
    max_tail_cate_len = max(tail_cate_len)
    max_head_senti_len = max(head_senti_len)
    max_tail_senti_len = max(tail_senti_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_entity_list = torch.LongTensor(cur_batch, entity_list[0].shape[0], max_entity_length, 2).zero_()
    # batch_size, relation_category, ....
    batch_head_cate_list = torch.LongTensor(cur_batch, head_cate_list[0].shape[0], max_head_cate_len, 2).zero_()
    batch_tail_cate_list = torch.LongTensor(cur_batch, tail_cate_list[0].shape[0], max_tail_cate_len, 2).zero_()

    batch_head_senti_list = torch.LongTensor(cur_batch, head_senti_list[0].shape[0], max_head_senti_len, 2).zero_()
    batch_tail_senti_list = torch.LongTensor(cur_batch, tail_senti_list[0].shape[0], max_tail_senti_len, 2).zero_()

    for i in range(cur_batch):
        # input_ids复制到batch_input_ids，长度对不上的保持原来的为0
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(attention_mask[i]))

        # entity_list shape：2 * ? * 2  因为原entity_list的长度为length[i]，所以复制的时候其中第一个维度只需要复制:length[i]
        batch_entity_list[i, :, :entity_length[i], :].copy_(torch.from_numpy(entity_list[i]))
        batch_head_cate_list[i, :, :head_cate_len[i], :].copy_(torch.from_numpy(head_cate_list[i]))
        batch_tail_cate_list[i, :, :tail_cate_len[i], :].copy_(torch.from_numpy(tail_cate_list[i]))
        batch_head_senti_list[i, :, :head_senti_len[i], :].copy_(torch.from_numpy(head_senti_list[i]))
        batch_tail_senti_list[i, :, :tail_senti_len[i], :].copy_(torch.from_numpy(tail_senti_list[i]))

    return {"text": text,  # tuple, shape: batch_size
            "input_ids": batch_input_ids,  # tuple, shape: batch_size without padding
            "attention_mask": batch_mask,  # tuple, shape: batch_size without padding
            "entity_list": batch_entity_list,  # Tensor, shape: batch_size * 2 * 一批当中entity_list的最大长度 * 2
            "head_cate_list": batch_head_cate_list,
            "tail_cate_list": batch_tail_cate_list,  # Tensor, shape: batch_size * 关系种类 * 一批当中tail_list的最大长度 * 2
            "head_senti_list": batch_head_senti_list,
            "tail_senti_list": batch_tail_senti_list,  # Tensor, shape: batch_size * 关系种类 * 一批当中tail_list的最大长度 * 2
            "quadruple_list": spo_list,  # tuple, shape: batch_size
            "token": token}


if __name__ == '__main__':
    from config import Config
    from torch.utils.data import DataLoader

    config = Config()
    dataset = MyDataset(config, config.train_data)
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)
    for data in dataloader:
        # print("*" * 50)
        print(data["entity_list"].shape)
        print(data["head_list"].shape)
        print(data["tail_list"].shape)
