import os.path

import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel, BertModel, GPT2LMHeadModel
from transformers import BertTokenizer, AutoModelForMaskedLM
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math
class MyOwnEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
        # self.dic = {'[PAD]': self.bart_encoder.embed_tokens.weight.data[0],
        #             '[UNK]': self.bart_encoder.embed_tokens.weight.data[100],
        #             '[CLS]': self.bart_encoder.embed_tokens.weight.data[101],
        #             '[SEP]': self.bart_encoder.embed_tokens.weight.data[102],
        #             '[unused1]': self.bart_encoder.embed_tokens.weight.data[1],
        #             '[unused2]': self.bart_encoder.embed_tokens.weight.data[2],
        #             '[unused3]': self.bart_encoder.embed_tokens.weight.data[3]}
        # self.bart_encoder.embed_tokens.weight.data[0] = self.dic['[CLS]']
        # self.bart_encoder.embed_tokens.weight.data[1] = self.dic['[PAD]']
        # self.bart_encoder.embed_tokens.weight.data[2] = self.dic['[SEP]']
        # self.bart_encoder.embed_tokens.weight.data[3] = self.dic['[UNK]']
        # self.bart_encoder.embed_tokens.weight.data[100] = self.dic['[unused1]']
        # self.bart_encoder.embed_tokens.weight.data[101] = self.dic['[unused2]']
        # self.bart_encoder.embed_tokens.weight.data[102] = self.dic['[unused3]']



    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder
        self.dic = {'[PAD]': self.bart_encoder.embed_tokens.weight.data[0],
                    '[UNK]': self.bart_encoder.embed_tokens.weight.data[100],
                    '[CLS]': self.bart_encoder.embed_tokens.weight.data[101],
                    '[SEP]': self.bart_encoder.embed_tokens.weight.data[102],
                    '[unused1]': self.bart_encoder.embed_tokens.weight.data[1],
                    '[unused2]': self.bart_encoder.embed_tokens.weight.data[2],
                    '[unused3]': self.bart_encoder.embed_tokens.weight.data[3]}
        self.bart_encoder.embed_tokens.weight.data[0] = self.dic['[CLS]']
        self.bart_encoder.embed_tokens.weight.data[1] = self.dic['[PAD]']
        self.bart_encoder.embed_tokens.weight.data[2] = self.dic['[SEP]']
        self.bart_encoder.embed_tokens.weight.data[3] = self.dic['[UNK]']
        self.bart_encoder.embed_tokens.weight.data[100] = self.dic['[unused1]']
        self.bart_encoder.embed_tokens.weight.data[101] = self.dic['[unused2]']
        self.bart_encoder.embed_tokens.weight.data[102] = self.dic['[unused3]']

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        self.dic = {'[PAD]': self.decoder.embed_tokens.weight.data[0],
                    '[UNK]': self.decoder.embed_tokens.weight.data[100],
                    '[CLS]': self.decoder.embed_tokens.weight.data[101],
                    '[SEP]': self.decoder.embed_tokens.weight.data[102],
                    '[unused1]': self.decoder.embed_tokens.weight.data[1],
                    '[unused2]': self.decoder.embed_tokens.weight.data[2],
                    '[unused3]': self.decoder.embed_tokens.weight.data[3]}
        self.decoder.embed_tokens.weight.data[0] = self.dic['[CLS]']
        self.decoder.embed_tokens.weight.data[1] = self.dic['[PAD]']
        self.decoder.embed_tokens.weight.data[2] = self.dic['[SEP]']
        self.decoder.embed_tokens.weight.data[3] = self.dic['[UNK]']
        self.decoder.embed_tokens.weight.data[100] = self.dic['[unused1]']
        self.decoder.embed_tokens.weight.data[101] = self.dic['[unused2]']
        self.decoder.embed_tokens.weight.data[102] = self.dic['[unused3]']
        # causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = torch.zeros(1024, 1024).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        # label_ids = sorted(label_ids, reverse=False)
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids) + 1
        # 这里在pipe中设置了第0个位置是sos, 第一个位置是eos, 所以做一下映射。还需要把task_id给映射一下
        mapping = torch.LongTensor([0, 2] + label_ids)
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask
        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                use_pos_cache=tokens.size(1) > 2)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1), self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[
                                            self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores

        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return self(tokens, state)[:, -1]


class CaGFBartDecoder(FBartDecoder):
    # Copy and generate, 计算score的时候，同时加上直接生成的分数
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=False):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        # 否则是计算完分数，分数平均
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, tokens, state):
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)  # eq判断元素是否相等， flip按维度反转， cumsum按维度累计求和
        """
            做预测时，最后一个字符为eos，但是传进来的数据，为了保持维度一样会填充eos
            例如：
            [101, 2, 2, 45, 67, 88, 99, 102, 102, 102, 102]
            tgt_pad_mask效果是只预测到第一个出现的eos，即102，因此，得到的tgt_pad_mask为：
            [False, False, False, False, False, False, False, False, True, True, True]
       """
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])  # target_token_pad_mask 第一个1是不需要mask，因为第一个1不属于pad, pipe中设置的tgt_token的padding为1

        # 把输入做一下映射
        """
        英文模型中，sos和eos在0，2的位置，1~6或者1~8位置都是直接在tokenizer中
        因此使用lt方法，可以直接比较大小，将小于src_start_index的mask
        但是中文模型中[cls]和[sep]符号位置不在0，2
        """
        mapping_token_mask = tokens.lt(self.src_start_index)  # 找出所有的特殊标记符号，例如“OE”，“AE”，“NEG”等
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)   # 得到的张量其中所有的真正src_token全部变成0， ge函数，逐个比较，不相等为0
        tag_mapped_tokens = self.mapping[mapped_tokens]  # 将向量中所有的特殊标记符变成0，属于src_token的变成tokenizer序列化的ids（在token中原来只是每个词在句子中的位置）

        src_tokens_index = tokens - self.src_start_index  # 将 target_tokens中所有特殊标记符的位置全变成小于0，方便下面过滤 # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)  # 所有特殊标记符全变成0，此时不为0的id都是表示在原有文本中token的位置
        src_tokens = state.src_tokens  # src_token中以cls即0开头，以sep即2结尾，以pad即1填充，state包含了四个特殊变量 encoder_mask, encoder_output, src_embed_outputs, src_tokens，其中src_token:batch_size, seq_len, 个人推测src_embed_outputs(batch_size, seq_len, 768)应该是在encoder最开始的嵌入层向量
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)  # gather按src_tokens_index取元素,如果src_tokens_index为[0, 6]，则去src_tokens中第1个和第7个，这一步是将target_tokens中除了特殊标记符号，都转换成tokenizer对应的id

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len  ， 这一步将target_token全部转换成tokenizer对应的id
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  # 已经映射之后的分数  target_tokens中，全部用tokenizer的id代替（不再是代表位置），此外eos用 2 (<SEP>) 表示，padding用 1 (<pad>) 表示
        # 至此，以上代码部分是为了传进来的traget_token变成使用tokenizer分词后的ids（原先target_token只是包含了特殊标记符在mapping中的位置，和token在src中的位置）
        if self.training:
            tokens = tokens[:, :-1]  # 参考decoder部分，最后一个是不作为输入的，因为前面加了个cls
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                use_pos_cache=tokens.size(1) > 3)
        hidden_state = dict.last_hidden_state  # batch_size, max_len ， hidden_size  max_len应该是根据target_tokens来计算的
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:  # 这个跟训练策略有关，减少了计算量，不用管
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(   # logits 形状为（x,x,x）, 与hidden_state具有相同的device,require_grad,dtype,数值均为 fill_value,大大简化了操作，最后一个维度设置的原因：可能是做判断，判断是哪一个位置的元素
            (hidden_state.size(0), hidden_state.size(1), self.src_start_index + src_tokens.size(-1)),
            fill_value=-1e24)

        # decoder输出的不含有<s>了，是包含</s>的，因此是要计算eos的位置分数
        eos_scores = F.linear(hidden_state,
                              self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # (bsz, max_len, 768) * (768, 1)  =>  bsz, max_len, 1 这应该是eos符号的分数， 因为用的是eos的向量同hidden_state计算的，下同
        tag_scores = F.linear(hidden_state, self.dropout_layer(
            self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x (21131-21128) # 这应该是三个特殊标记符（“AOESC”，“NEG”，“POS”）的分数

        # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来

        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output  # encoder的输出 bsz, max_len, 768
        if hasattr(self, 'encoder_mlp'):  # 用来判断对象是否包含此属性，这个应该就是映射函数，压缩维度的，或者就是一个线性变换的
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)  # bsz, 1, src_len(不是target_len), 这个是有关src_token的mask
        input_embed = self.decoder.embed_tokens(src_tokens)  # bsz ，src_len ， hidden_size
        input_embed = self.dropout_layer(input_embed)
        if self.avg_feature:  # 先把feature合并一下，encoder的输出和encoder的输入合并
            src_outputs = (src_outputs + input_embed) / 2   # src_outputs就是encoder的输出，bsz ，src_len ， hidden_size
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # hidden_state（bsz, traget_len-1, 768）是decoder的输出，src_outputs:(bsz, src_len, 768) ====> bsz , target_len-1 , src_len, 可以理解成原句子每个词在生成序列中每个位置的分数
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz ，target_len-1 ，src_len
            word_scores = (gen_scores + word_scores) / 2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))  # bsz, 1, src_len  , ge比较是否相同， 最后mask中eos和pad位置全为True，其余false
        word_scores = word_scores.masked_fill(mask, -1e32)  # 遮住就不计算pad在生成序列中位置的分数了，可能也不计算eos了

        logits[:, :, 1:2] = eos_scores  # 计算 eos分数： logits(bsz, traget_len-1, len(label_ids) + 2 + src_len)， 因为要算上eos 和sos 。 (bsz, target_len-1, 1)
        logits[:, :, 2:self.src_start_index] = tag_scores  # 计算 特殊标志符位置分数
        # logits[:, :, 2:4].fill_(-1e32)  # 直接吧task的位置设置为不需要吧
        logits[:, :, self.src_start_index:] = word_scores  # 最后计算src_token在target_token中位置的分数，logits[:, :, 0]应该是不需要考虑？？eos就是直接代表着结束？


        return logits  # 在后面计算loss的时候，第一个位置就没计算了。总的来说就是除了cls这个标签以外，返回原src中token在生成序列各位置的分数 （bsz, target_len-1, src_len）


class Restricter(nn.Module):
    def __init__(self, label_ids):
        super().__init__()
        self.src_start_index = 2 + len(label_ids)
        self.tag_tokens = label_ids

    def __call__(self, state, tokens, scores, num_beams=1):
        """

        :param state: 各种DecoderState
        :param tokens: bsz x max_len，基于这些token，生成了scores的分数
        :param scores: bsz x vocab_size*real_num_beams  # 已经log_softmax后的
        :param int num_beams: 返回的分数和token的shape
        :return:
        num_beams==1:
            scores: bsz x 1, tokens: bsz x 1
        num_beams>1:
            scores: bsz x num_beams, tokens: bsz x num_beams  # 其中scores是从大到小排好序的
        """
        bsz, max_len = tokens.size()
        logits = scores.clone()
        if max_len > 1 and max_len % 5 == 1:  # 只能是tags
            logits[:, :2].fill_(-1e24)
            logits[:, self.src_start_index:].fill_(-1e24)
        elif max_len % 5 == 2:
            logits[:, 2:self.src_start_index].fill_(-1e24)
        else:
            logits[:, :self.src_start_index].fill_(-1e24)

        _, ids = torch.topk(logits, num_beams, dim=1, largest=True, sorted=True)  # (bsz, num_beams)
        next_scores = scores.gather(index=ids, dim=1)

        return next_scores, ids


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None, copy_gate=False,
                    use_encoder_mlp=False, use_recur_pos=False, tag_first=False):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens) + num_tokens)

        encoder = model.encoder
        decoder = model.decoder
        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        tokenizer_path = os.path.abspath("../../../") + r"/resource" + "/" + "bart-base-chinese/vocab.txt"
        _tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        # 为一些特殊字符串设定向量[<<opinion_extraction>>, <<aspect_extraction>>, <<positive>>]
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                # indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))  #上一行
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)
        if decoder_type is None:
            assert copy_gate is False
            raise RuntimeError("Potential bug for this choice.")
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type == 'avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=False, use_encoder_mlp=use_encoder_mlp)
        elif decoder_type == 'avg_feature':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=True, use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
        """
        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, first, tgt_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new
