from pipe import WangBartABSAPipe
import sys
import os

sys.path.append('../')
import torch
from fastNLP import cache_results
import numpy as np
from itertools import chain
from copy import deepcopy

# 这一段代码的目的是为了加载cache好的东西，因为里面包含了tokenizer，mappping2id什么的，如果是单独额外保存的话，可以用其他方式加载的
dataset_name = 'wenqiChen1'
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

model = torch.load('model_results/best_SequenceGeneratorModel_F1_2024-03-26-11-05-06-603603')


def predict(sentence):
    # target decode这边的prompt
    aoesc_target = torch.LongTensor([0, mapping2targetid['E_R_E'], mapping2targetid['E_R_E']])
    eos_token_id = 1
    target_shift = len(mapping2id) + 2
    labels = list(mapping2id.keys())
    word_start_index = len(labels) + 2
    # 加载model

    # model.generator.set_new_generator() # 可以重新设置一些与生成相关的参数

    # 准备数据
    sents_tmp = [sentence]
    # sents_tmp = ['医生的素质不行啊，效果还勉强说的过去']
    sents = [' '.join([i for i in sents_tmp[0]])]

    # 数据准备
    src_tokens = []
    src_seq_len = []
    mappingbacks = []
    aoesc_target_tokens = []
    raw_words = []
    for sent in sents:
        _raw_words = sent.split()
        raw_words.append(_raw_words)
        word_bpes = [[tokenizer.bos_token_id]]
        for word in _raw_words:
            bpes = tokenizer.tokenize(word)
            bpes = tokenizer.convert_tokens_to_ids(bpes)
            word_bpes.append(bpes)
        word_bpes.append([tokenizer.eos_token_id])
        lens = list(map(len, word_bpes))
        cum_lens = np.cumsum(list(lens)).tolist()
        mappingback = np.full(shape=(cum_lens[-1] + 1), fill_value=-1, dtype=int).tolist()
        for _i, _j in enumerate(cum_lens):
            mappingback[_j] = _i
        mappingbacks.append(mappingback)
        src_tokens.append(torch.LongTensor(list(chain(*word_bpes))))
        src_seq_len.append(len(src_tokens[-1]))
        aoesc_target_tokens.append(aoesc_target)
    encoder_inputs = {'src_tokens': torch.nn.utils.rnn.pad_sequence(src_tokens, batch_first=True,
                                                                    padding_value=tokenizer.pad_token_id),
                      'src_seq_len': torch.LongTensor(src_seq_len)}

    # 因为是共享encode的，所以单独拆分成encode和generate
    model.eval()
    with torch.no_grad():
        state = model.seq2seq_model.prepare_state(**encoder_inputs)
        # print(state.num_samples)
        aoesc_result = model.generator.generate(deepcopy(state), tokens=torch.stack(aoesc_target_tokens,
                                                                                    dim=0))  # the prompt is provided to the model

    # 抽取aesc数据
    aspects = []
    opinions = []
    aoesc_eos_index = aoesc_result.flip(dims=[1]).eq(eos_token_id).cumsum(dim=1).long()
    aoesc_result = aoesc_result[:, 1:]
    aoesc_seq_len = aoesc_eos_index.flip(dims=[1]).eq(aoesc_eos_index[:, -1:]).sum(dim=1)  # bsz
    aoesc_seq_len = (aoesc_seq_len - 2).tolist()
    for i, (ps, length, mappingback, _raw_words) in enumerate(zip(aoesc_result.tolist(), aoesc_seq_len,
                                                                  mappingbacks, raw_words)):
        ps = ps[2:length]
        pairs = []
        cur_pair = []  # each pair with the format (a_start, a_end, class), start/end inclusive, and considering the sos in the start of sentence
        if len(ps):
            for index, j in enumerate(ps):
                if j < word_start_index:
                    cur_pair.append(j)
                    if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                        pass
                    else:
                        pairs.append(tuple(cur_pair))
                    cur_pair = []
                else:
                    cur_pair.append(j)
        _aspects = []
        _opinions = []
        for index, pair in enumerate(pairs):
            a_s, a_e, o_s, o_e, sc = pair
            a_s, a_e, o_s, o_e = mappingback[a_s - target_shift], mappingback[a_e - target_shift] + 1, mappingback[o_s - target_shift], mappingback[o_e - target_shift] + 1
            _aspects.append({
                'index': index,
                'from': a_s,
                'to': a_e,
                'polarity': labels[sc - 2].upper(),
                'term': ' '.join(_raw_words[a_s:a_e])
            })
            _opinions.append({
                'index': index,
                'from': o_s,
                'to': o_e,
                'polarity': labels[sc - 2].upper(),
                'term': ' '.join(_raw_words[o_s:o_e])
            })
        aspects.append(_aspects)
        opinions.append(_opinions)

    for k, v in zip(aspects[0], opinions[0]):
        print('<', k['term'], ',', v['term'], ',', k['polarity'], '>')
    # print(aspects)
    #
    # print(opinions)

if __name__ == '__main__':

    while True:
        input_seq = input("输入:")

        predict(input_seq)
