# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/31 10:30
@Auth ： 叶东宇
@File ：tester.py
"""
from pipe import WangBartABSAPipe
import sys

sys.path.append('../')
from model.bart.metrics import TripletSpanMetric
import os

import torch
from fastNLP import cache_results
from fastNLP import Tester

# 这一段代码的目的是为了加载cache好的东西，因为里面包含了tokenizer，mappping2id什么的，如果是单独额外保存的话，可以用其他方式加载的
# 这一段代码的目的是为了加载cache好的东西，因为里面包含了tokenizer，mappping2id什么的，如果是单独额外保存的话，可以用其他方式加载的
dataset_name = 'chip2020'
opinion_first = False
cache_fn = f"caches/data_{dataset_name}.pt"


@cache_results(cache_fn, _refresh=False)
def get_data():
    print("正在创建数据流")
    pipe = WangBartABSAPipe(opinion_first=opinion_first)
    data_path = os.path.abspath("../../../") + r"/data" + "/" + dataset_name
    data_bundle, max_len, max_len_a = pipe.process_from_file(data_path, demo=False)
    return data_bundle, max_len, max_len_a, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid


data_bundle, max_len, max_len_a, tokenizer, mapping2id, mapping2targetid = get_data()
conflict_id = -1 if 'CON' not in mapping2targetid else mapping2targetid['CON']
label_ids = list(mapping2id.values())

model = torch.load('model_results/best_SequenceGeneratorModel_F1_2024-03-18-07-38-24-583995')
tester = Tester(data_bundle.get_dataset("devE_R_E"),
                model,
                TripletSpanMetric(1, num_labels=len(label_ids), conflict_id=conflict_id),
                batch_size=24 * 5,
                num_workers=0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=0,
                use_tqdm=False,
                fp16=False
                )
print(tester.test())
