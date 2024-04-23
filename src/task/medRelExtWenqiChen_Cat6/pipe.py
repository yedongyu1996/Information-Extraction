from fastNLP.io import Pipe, DataBundle, Loader
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer, BertTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key
from tqdm import tqdm
import sys
sys.path.append("../..")
from model.utils import get_max_len_max_len_a
import os




def cmp(v1, v2):
    if v1['from']==v2['from']:
        return v1['to'] - v2['to']
    return v1['from'] - v2['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


def dataset_detail(data_path):
    max_len = 0  # 最大生成长度
    total_data = 0  # 总数据量
    relation_kind_total = {}  # 记录所有数据集下每种关系实体的数量
    triplet_statistic_total = {}  # 记录所有数据中三元组数量的分布
    gold_num = {"dev_convert": 0, "test_convert": 0, "train_convert": 0}
    for path in ['dev_convert', 'test_convert', 'train_convert']:
        with open(data_path + '/' + path + '.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(path.replace("_convert",''),' 数据量:  ', len(data))
        total_data += len(data)
        length = 0
        triplet_statistic = {}  # 记录每个数据集中三元组数量的分布
        relation_kind = {}  # 记录每个数据集下每种关系实体的数量
        raw_length = {}  # 原数据长度
        for ins in data:
            aspects = ins['aspects']
            length += len(aspects)
        for ins in data:
            gold_num[path] += len(ins["aspects"])
            for item in ins["aspects"]:
                rels = item["polarity"]
                if rels not in relation_kind:
                    relation_kind[rels] = 1
                else:
                    relation_kind[rels] += 1

                if rels not in relation_kind_total:
                    relation_kind_total[rels] = 1
                else:
                    relation_kind_total[rels] += 1

            raw_words_length = len(ins["words"])
            if raw_words_length not in raw_length:
                raw_length[raw_words_length] = 1
            else:
                raw_length[raw_words_length] += 1

            aspects = ins['aspects']
            if len(aspects) not in triplet_statistic:
                triplet_statistic[len(aspects)] = 1
            else:
                triplet_statistic[len(aspects)] += 1

            if len(aspects) not in triplet_statistic_total:
                triplet_statistic_total[len(aspects)] = 1
            else:
                triplet_statistic_total[len(aspects)] += 1

            tmp_len = len(aspects)*3 + 4
            if tmp_len > max_len:
                max_len = tmp_len

        print("========关系数量的分布 [实体关系:数量]: ", sorted(relation_kind.items(), key=lambda x: x[0]))
        print("========三元组分布情况 [三元组个数:数量]: ", sorted(triplet_statistic.items(), key=lambda x: x[0]))
        print("========三元组个数综合: ", sum([v for k, v in relation_kind.items()]))
        print("========原文本长度分布 [文本长度:数量]: ", sorted(raw_length.items(), key=lambda x: x[0]))
    print('数据集总量: ', total_data)
    print("========关系数量的分布 [实体关系:数量]: ", sorted(relation_kind_total.items(), key=lambda x: x[0]))
    print("========总三元组分布[三元组个数:数量]: ", sorted(triplet_statistic_total.items(), key=lambda x: x[0]))
    print("gold_num: ", gold_num)

    return max_len

class WangBartABSAPipe(Pipe):
    def __init__(self, opinion_first=False):
        super(WangBartABSAPipe, self).__init__()

        tokenizer_path = os.path.abspath('../../../') + r"/resource/bart-base-chinese/vocab.txt"
        self.tokenizer = BertTokenizer(tokenizer_path)

        """
        self.mapping中value应该全部小写，不能大写
        """
        self.mapping = {  # 应该是要在他们的前面，
            "E_R_E": "<<entity_relation_entity_extraction>>",
            '疾病-属性': "<<疾病-属性>>",
            '部位-疾病': "<<部位-疾病>>",
            '部位-属性': "<<部位-属性>>",
            '部位-征象': "<<部位-征象>>",
            '部位-症状': "<<部位-症状>>",
            '征象-属性': "<<征象-属性>>"
        }
        self.opinion_first = opinion_first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_tokens = cur_num_tokens

    def add_tokens(self):
        """
        添加的tokens最好全部小写的英文，否则可能会找不到
        """
        # tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        # sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        sorted_add_tokens = list(self.mapping.values())
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        # print("===================",self.tokenizer.tokenize())
        self.mapping2id = {}  # 将mapping中的符号转换成tokenizer中的id
        self.mapping2targetid = {}  # 将mapping中符号转换成target_token中的id符号，即任务符号和情感符号，（加上2以后）

        # 需要保证Aspect_Opinion_extraction_Sentiment_Analysis是第一位的
        for i, value in enumerate(
                [
                    "<<entity_relation_entity_extraction>>",
                    "<<疾病-属性>>",
                    "<<部位-疾病>>",
                    "<<部位-属性>>",
                    "<<部位-征象>>",
                    "<<部位-症状>>",
                    "<<征象-属性>>"
                ]
        ):
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))[0]
            assert key_id == self.cur_num_tokens+i

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid) + 2

    def process(self, data_bundle: DataBundle) -> DataBundle:
        self.add_tokens()
        target_shift = len(self.mapping) + 2  # 还包括其实的ClS和结尾的SEP
        for name in ['train', 'dev', 'test']:
            ds = data_bundle.get_dataset(name)
            tmp_ds = DataSet()
            if name == 'train':  # 训练时，所有任务在一起训练
                aspect_opinion_sentiment_ds = tmp_ds
            else:  # 测试和验证时，每一种任务单独执行
                aspect_opinion_sentiment_ds = DataSet()
            print("处理"+name+"数据集: ")
            for ins in tqdm(ds):
                raw_words = ins['raw_words']
                word_to_TokenizerIds = [[self.tokenizer.bos_token_id]]
                for word in raw_words:
                    token_list = self.tokenizer.tokenize(word)
                    tokens_to_ids = self.tokenizer.convert_tokens_to_ids(token_list)
                    word_to_TokenizerIds.append(tokens_to_ids)
                word_to_TokenizerIds.append([self.tokenizer.eos_token_id])   # 输入进encoder的数据是有cls和sep以及中间的数据组成的。

                lens = list(map(len, word_to_TokenizerIds))
                cum_lens = np.cumsum(list(lens)).tolist()

                aspect_opinion_sentiment_target = [0, self.mapping2targetid['E_R_E'], self.mapping2targetid['E_R_E']]

                entity_relation_entity_span = []

                _word_to_TokenizerIds = list(chain(*word_to_TokenizerIds))  # 同前面分词和转换成id有关，不需要管
                aspects = sorted(ins['aspects'], key=cmp_to_key(cmp))
                opinions = sorted(ins['opinions'], key=cmp_to_key(cmp))
                for i in range(len(aspects)):
                    """
                    aspect和opinion的spans
                    """
                    aspect_start_ids = cum_lens[aspects[i]['from']] + target_shift
                    aspect_end_ids = cum_lens[aspects[i]['to'] - 1] + target_shift
                    opinion_start_ids = cum_lens[opinions[i]['from']] + target_shift
                    opinion_end_ids = cum_lens[opinions[i]['to'] - 1] + target_shift

                    """
                    sentiment polarity
                    """
                    polarity = self.mapping2targetid[aspects[i]['polarity']]
                    """
                    开始存储各类任务的数据
                    """
                    entity_relation_entity_span.append(
                        (aspect_start_ids, aspect_end_ids, opinion_start_ids, opinion_end_ids, polarity)
                    )  # A_O_S span
                    aspect_opinion_sentiment_target.extend(
                        [aspect_start_ids, aspect_end_ids, opinion_start_ids, opinion_end_ids, polarity]
                    )  # A_O_S target

                """
                 生成的目标数据形式<s> <任务类型> <任务类型> <目标数据> </s> ，
                 这也是为什么前面在add_tokens方法中最后一行len(self.mapping2targetid) + 2
                 因为加了个起始和结束的符号<s> </s>
                """
                aspect_opinion_sentiment_target.append(1)
                aspect_opinion_sentiment_ins = Instance(
                    src_tokens=_word_to_TokenizerIds.copy(), tgt_tokens=aspect_opinion_sentiment_target
                )
                """
                train的时候，只计算在decoder的输入和decoder输出的差别
                dev和test时候：才会计算抽取的r、p、f1
                """
                if name != 'train':
                    aspect_opinion_sentiment_ins.add_field('entity_relation_entity_span', entity_relation_entity_span)

                aspect_opinion_sentiment_ds.append(aspect_opinion_sentiment_ins)

            if name == 'train':
                data_bundle.set_dataset(tmp_ds, 'train')
            else:
                data_bundle.set_dataset(aspect_opinion_sentiment_ds, name + 'E_R_E')

        data_bundle.set_ignore_type(
            "entity_relation_entity_span"
        )
        data_bundle.set_pad_val('tgt_tokens', 1)  # padding值设置为1
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)  # padding值设置为0

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target(  # input属性设置为true
            'tgt_tokens',
            'tgt_seq_len',
            "entity_relation_entity_span"
        )

        return data_bundle

    def process_from_file(self, paths, demo=False):
        max_len = dataset_detail(paths)
        data_bundle = ABSALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)
        a,max_len_a = get_max_len_max_len_a(data_bundle, max_len)

        return data_bundle, max_len, max_len_a


class ABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        delete = 0
        for ins in data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            if len(opinions)==1:
                if len(opinions[0]['term'])==0:
                    opinions = []
            if len(aspects)==1:
                if len(aspects[0]['term'])==0:
                    aspects = []
            new_aspects = []
            for aspect in aspects:
                if 'polarity' not in aspect:
                    delete += 1
                    continue
                new_aspects.append(aspect)

            ins = Instance(raw_words=tokens, aspects=new_aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds) > 30:
                break
        return ds



if __name__ == '__main__':

    tmp = WangBartABSAPipe()
    dataset = "food"
    dataset_path = os.path.abspath('../../../') + r"/data" + "/" + dataset

    # max_len = dataset_detail(dataset_path)
    # print(max_len)
    data_bundle = tmp.process_from_file(dataset_path)

    #
    # pipe = WangBartABSAPipe(tokenizer='vocab.txt')
    # print(pipe.tokenizer.pad_token_id)
    # print(pipe.tokenizer.convert_tokens_to_ids('[PAD]'))


