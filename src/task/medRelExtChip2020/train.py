import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
import warnings
warnings.filterwarnings('ignore')
from pipe import WangBartABSAPipe
import sys
sys.path.append("../..")
from model.bart.bart_wang import BartSeq2SeqModel, Restricter
from model.bart.metrics import TripletSpanMetric
from model.bart.losses import Seq2SeqLoss
from model.bart.callbacks import FitlogCallback, WarmupCallback
from model.bart.generater import SequenceGeneratorModel
import torch
from fastNLP import Trainer, Tester
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results
import fitlog
from fastNLP.core.sampler import SortedSampler
import numpy as np
import random
import argparse


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    if 'p' not in os.environ and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        device = 'cuda:0'
    else:
        device = 'cuda'
else:
    device = 'cpu'
fitlog.set_log_dir('logs')
fitlog.add_hyper_in_file(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='chip2020', type=str)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_beams', default=1, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=1000, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score', 'avg_feature'])
parser.add_argument('--length_penalty', default=1.0, type=float)
pretrain_model = "bart-base-chinese"
pretrain_model_dir = os.path.abspath('../../../') + r"/resource" + "/" + pretrain_model
parser.add_argument('--bart_name', default=pretrain_model_dir, type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--warmup', type=float, default=0.1)
# parser.add_argument('--use_encoder_mlp', action='store_true', default=False)
args = parser.parse_args()
lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
warmup = args.warmup
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
fitlog.add_hyper(args)
use_encoder_mlp = args.use_encoder_mlp

demo = False
if demo:
    cache_fn = f"caches/data_{dataset_name}_demo.pt"
else:
    cache_fn = f"caches/data_{dataset_name}.pt"


# 这里生成的数据，是没有first而是直接bpe的
@cache_results(cache_fn, _refresh=False)
def get_data():
    print("正在创建数据流")
    pipe = WangBartABSAPipe(opinion_first=opinion_first)
    data_path = os.path.abspath("../../../") + r"/data" + "/" + dataset_name
    data_bundle, max_len, max_len_a = pipe.process_from_file(data_path, demo=demo)
    return data_bundle, max_len, max_len_a, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid

data_bundle, max_len, max_len_a, tokenizer, mapping2id, mapping2targetid = get_data()

conflict_id = -1 if 'CON' not in mapping2targetid else mapping2targetid['CON']
# print(data_bundle)
max_len = 40

"""
在utils.py文件
get_max_len_max_len_a方法中确定
"""
max_len_a = {
    'medical': 0.0,
    'medical2': 0.0,
    'food': 0.6,
    'MedRelChip2020': 0.6,
    'chip2020': 0.6
}[dataset_name]

label_ids = list(mapping2id.values())

"""
BartSeq2SeqModel是一个通用的encoder2decoder架构
encoder负责输入和部分输出
decoder将encoder的部分输出作为输入，输出全部的序列

train时，需要BartSeq2SeqModel作为模型，
predict时，需要SequenceGeneratorModel来生成

训练的时候decoder端是需要将target_token作为输出的
也就是用上一个时刻的输出作为输入，去预测当前时刻的输出是不是下一个时刻的输入
所以在bart_wang.py中line 375，训练时需要将target_token作为输入
测试的时候
但是预测的时候是不给target_token的
预测仅指的是predict
"""
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
restricter = Restricter(label_ids)
bos_token_id = 0  # 因为是特殊符号
eos_token_id = 1  # 因为是特殊符号 TODO 特别需要注意这是1, 因为把这些token都做了重映射

"""
SequenceGeneratorModel是生成式架构
参数是需要BartSeqSeqModel这样类型的实例作为参数
"""
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a, num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None)
"""
以下处理模型参数的学习率
"""
parameters = []
params = {'lr': lr, 'weight_decay': 1e-2}
params['params'] = [param for name, param in model.named_parameters() if
                    not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr': lr, 'weight_decay': 1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr': lr, 'weight_decay': 0}

params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)

callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=warmup, schedule='linear'))
callbacks.append(FitlogCallback(tester={
    'testE_R_E': Tester(data=data_bundle.get_dataset('testE_R_E'), model=model,
                        metrics=TripletSpanMetric(eos_token_id, num_labels=len(label_ids), conflict_id=conflict_id),
                        batch_size=batch_size * 5, num_workers=0, device=device, verbose=0, use_tqdm=False,
                        fp16=False)
}))

dev_data = data_bundle.get_dataset('devE_R_E')
# dev_data = None

sampler = None
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = [TripletSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=False, conflict_id=conflict_id)]

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=0, n_epochs=n_epochs, print_every=1 if 'SEARCH_OUTPUT_FP' not in os.environ else 100,
                  dev_data=dev_data, metrics=metric, metric_key='F1',
                  validate_every=-1, save_path='model_results', use_tqdm='SEARCH_ID' not in os.environ, device=device,
                  callbacks=callbacks, check_code_level=-1 if 'SEARCH_ID' in os.environ else 0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size * 5)

trainer.train(load_best_model=False)

if trainer.save_path is not None:
    model_name = "best_" + "_".join([model.__class__.__name__, trainer.metric_key, trainer.start_time])
    fitlog.add_other(name='model_name', value=model_name)