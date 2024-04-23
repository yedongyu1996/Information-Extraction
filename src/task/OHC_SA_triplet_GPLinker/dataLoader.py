import json
import os.path
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import torch


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

        # 对于文本中的每一个三元组，entity_list[0]存储实体1， entity_list[1]存储实体2
        # entity_list[0]    shape：n*2 n是数据集下所有三元组个数   2是记录了每个实体的头尾位置
        # entity_list[1]    shape：n*2 n是数据集下所有三元组个数   2是记录了每个实体的头尾位置
        # 由于存在重叠的现象，即多个实体与同一个实体有关系，因此这两个元素的长度不是总相同的，且每个元素的长度不一定同总共三元组的个数相同（有时是第一个实体重叠，有时是第二个实体重叠）
        # entity_list = [set() for _ in range(2)]
        entity_list = [[], []]  # 分别记录头尾

        # 记录与relation相关的实体的头位置  shape：n n是关系类别数，元素类型为set
        # 其中每个元素存储的是一个三元组中两个实体的头位置，即按关系类别存储涉及到的三元组实体头位置
        head_list = [[] for _ in range(self.config.num_rel)]

        # 记录与relation相关的实体的尾位置  shape：n n是关系类别数，元素类型为set
        # 其中每个元素存储的是一个三元组中两个实体的尾位置，即按关系类别存储涉及到的三元组实体尾位置
        tail_list = [[] for _ in range(self.config.num_rel)]  # 记录三元组中的关系涉及到的两个实体的尾位置，长度为关系的类别数

        triple_list = []

        for aspect, opinion in zip(json_data["aspects"], json_data["opinions"]):
            entity_list[0].append([aspect["from"] + 1, aspect["to"]])  # +1是因为前面加了CLS
            entity_list[1].append([opinion["from"] + 1, opinion["to"]])
            triple_list.append(
                [aspect["from"] + 1, aspect["to"], aspect["polarity"], opinion["from"] + 1, opinion["to"]])
            rel = self.label2id[aspect['polarity']]
            head_list[rel].append([aspect["from"] + 1, opinion["from"] + 1])
            tail_list[rel].append([aspect["to"], opinion["to"]])

        # 这三个数据中entity_list不太可能出现空数据的现象
        # 只有head_list和tail_list，因为一条样本可能不涉及某一种关系，因此某个元素可能为空，对词添加（0,0）
        # 添加过后不存在空数据了，出现（0,0）的数据只可能是因为不存在然后后续添加的
        for label in entity_list + head_list + tail_list:
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

        head_length = [len(i) for i in head_list]
        max_hl = max(head_length)
        tail_length = [len(i) for i in tail_list]
        max_tl = max(tail_length)

        # head_list转换成list，shape： n*m*l
        # n 关系种类；m 是max_hl，即head_list元素中最大长度；l 2，记录一个三元组中两个实体的头位置
        head_list = padding(head_list, max_hl, text)

        # tail_list转换成list，shape： n*m*l
        # n 关系种类；m 是max_hl，即tail_list元素中最大长度；l 2，记录一个三元组中两个实体的尾位置
        tail_list = padding(tail_list, max_tl, text)

        # entity_list = np.array([list(i) for i in entity_list])

        return text, token, triple_list, input_ids, attention_mask, token_len, entity_list_length, \
            entity_list, head_list, tail_list, max_hl, max_tl


def collate_fn(batch):
    text, token, spo_list, input_ids, attention_mask, token_len, length, entity_list, head_list, tail_list, head_len, tail_len = zip(
        *batch)

    cur_batch = len(batch)
    max_text_length = max(token_len)
    max_length = max(length)
    max_head_len = max(head_len)
    max_tail_len = max(tail_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_entity_list = torch.LongTensor(cur_batch, 2, max_length, 2).zero_()
    batch_head_list = torch.LongTensor(cur_batch, 6, max_head_len, 2).zero_()
    batch_tail_list = torch.LongTensor(cur_batch, 6, max_tail_len, 2).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(
            torch.from_numpy(input_ids[i]))  # input_ids复制到batch_input_ids，长度对不上的保持原来的为0
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(attention_mask[i]))

        # entity_list shape：2 * ? * 2  因为原entity_list的长度为length[i]，所以复制的时候其中第一个维度只需要复制:length[i]
        batch_entity_list[i, :, :length[i], :].copy_(torch.from_numpy(entity_list[i]))
        batch_head_list[i, :, :head_len[i], :].copy_(torch.from_numpy(head_list[i]))
        batch_tail_list[i, :, :tail_len[i], :].copy_(torch.from_numpy(tail_list[i]))

    return {"text": text,  # tuple, shape: batch_size
            "input_ids": batch_input_ids,  # tuple, shape: batch_size without padding
            "attention_mask": batch_mask,  # tuple, shape: batch_size without padding
            "entity_list": batch_entity_list,  # Tensor, shape: batch_size * 2 * 一批当中entity_list的最大长度 * 2
            "head_list": batch_head_list,
            # Tensor, shape: batch_size * 关系种类 * 一批当中head_list的最大长度 * 2   一个样本中，存在九种关系，head_list存储的是每一个三元组两个实体的头位置，因此最后一个维度是2；倒数第二个维度代表每种关系下存在多少个三元组。
            "tail_list": batch_tail_list,  # Tensor, shape: batch_size * 关系种类 * 一批当中tail_list的最大长度 * 2
            "triple_list": spo_list,  # tuple, shape: batch_size
            "token": token}


if __name__ == '__main__':
    from config import Config
    from torch.utils.data import DataLoader

    config = Config()
    config.dataset = "bart_format"
    dataset = MyDataset(config, config.train_data)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    for data in dataloader:
        # print("*" * 50)
        print(data["entity_list"].shape)
        print(data["head_list"].shape)
        print(data["tail_list"].shape)
