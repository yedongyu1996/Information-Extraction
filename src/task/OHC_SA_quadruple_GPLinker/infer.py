import sys
sys.path.append("../../model/GPLinker_quadruple")  # 添加模型的模块
sys.path.append("../../task/OHC_SA_quadruple_GPLinker")  # 添加任务的模块
import os
os.environ["project_root"] = os.path.abspath("../../..")  # 定位到项目根目录，以便后续加载文件
from framework import Framework  # 爆红不用管
from models import GlobalLinker
from config import Config
import torch
from transformers import BertTokenizer
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
model = GlobalLinker(config).to(device)
model.load_state_dict(torch.load('checkpoint/' + 'GPLinker_triplet_2024-04-24-21-36-36.pt'), strict=False)
model.eval()
threshold=-0.5
tokenizer = BertTokenizer.from_pretrained(os.environ["project_root"] + '/resource/bert-base-chinese/vocab.txt', do_lower_case=False)
with open(os.environ["project_root"] + "/" + config.map_cate, "r", encoding="utf-8") as f:
    cate_id2label = json.load(f)[1]
with open(os.environ["project_root"] + "/" + config.map_senti, "r", encoding="utf-8") as f:
    senti_id2label = json.load(f)[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
while True:
    print("输入文本：", end='')
    tmp_text = input().replace('\n', '').replace(' ', '')
    token = ['[CLS]']
    for i in range(len(tmp_text)):
        token.append(tmp_text[i])
    token.append('[SEP]')
    token2id = [tokenizer.convert_tokens_to_ids(token)]
    mask = [[1] * len(token)]
    attention_mask = np.array(mask)
    input_ids = np.array(token2id)
    data = {}
    data["input_ids"] = torch.from_numpy(input_ids).to(device)
    data["attention_mask"] = torch.from_numpy(attention_mask).to(device)
    with torch.no_grad():
        text = tmp_text
        logits = model(data)
        outputs = [o.cpu()[0] for o in logits]  # relation_logits, head_logits, tail_logits
        outputs[0][:, [0, -1]] -= np.inf  # 2 * max_length * 2， 数据处理时 存在[CLS] 和[SEP ], 这是行处理
        outputs[0][:, :, [0, -1]] -= np.inf  # 2 * max_length * 2， 数据处理时 存在[CLS] 和[SEP ], 这是列处理

        subjects, objects = [], []
        for l, h, t in zip(*np.where(outputs[0] > threshold)):
            if l == 0:
                subjects.append((h, t))
            else:
                objects.append((h, t))

        predict = []
        for sh, st in subjects:
            for oh, ot in objects:
                cate_head = np.where(outputs[1][:, sh, oh] > threshold)[0]
                cate_tail = np.where(outputs[2][:, st, ot] > threshold)[0]
                cate = set(cate_head) & set(cate_tail)  # cate
                for c in cate:
                    senti_head = np.where(outputs[3][:, sh, oh] > threshold)[0]
                    senti_tail = np.where(outputs[4][:, st, ot] > threshold)[0]
                    ss = set(senti_head) & set(senti_tail)  # cate
                    for s in ss:
                        category = cate_id2label[str(c)]
                        senti = senti_id2label[str(s)]
                        predict.append((token[sh: st+1], category, senti, token[oh: ot+1]))
        for vol in predict:
            print(vol)