import os

import torch
import torch.nn as nn
from transformers import BertModel


class GlobalPointer(nn.Module):
    """
    核心思想:
    使用GlobalPointer做分类，预测头实体和尾实体是否有关系，以及是什么关系，都用到了GlobalPointer
    最终的结果都是输出seq_len * seq_len
    也就是最终都是对序列中每个位置与另一个位置的关系预测
    并通过这个预测来解码实体与关系。
    """

    def __init__(self, config, num_type):
        super(GlobalPointer, self).__init__()
        self.config = config
        # self.RoPE = self.config.RoPE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.bert_dim = self.config.bert_dim
        self.num_type = num_type
        self.linear = nn.Linear(self.bert_dim, self.num_type * self.config.hidden_size * 2)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, bert_last_hidden_state, attention_mask):
        # input_ids = data["input_ids"].to(self.device)
        # attention_mask = data["mask"].to(self.device)

        # [batch_size, seq_len, bert_dim]
        last_hidden_state = bert_last_hidden_state
        batch_size = last_hidden_state.size(0)
        seq_len = last_hidden_state.size(1)

        # [batch_size, seq_len, num_type * hidden_size * 2]
        outputs = self.linear(last_hidden_state)
        # [[batch_size, seq_len, hidden_size * 2], [batch_size, seq_len, num_type * hidden_size * 2 - self.config.hidden_size * 2]]
        outputs = torch.split(outputs, self.config.hidden_size * 2,
                              dim=-1)  ##分割最后一个维度，先取出config.hidden_size * 2，然后将剩下的在放到另一个矩阵中，返回一个元组
        # [batch_size, seq_len, num_type, hidden_size * 2]
        outputs = torch.stack(outputs, dim=-2)  ##按维度堆叠
        # [batch_size, seq_len, num_type, hidden_size]
        qw, kw = outputs[..., :self.config.hidden_size], outputs[..., self.config.hidden_size:]

        if self.config.RoPE:
            # [batch_size, seq_len, hidden_size]
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.config.hidden_size)

            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # [batch_size, num_type, seq_len, seq_len]，这里的计算同hidden_size无关，反正最后得到的是一个seq_len * seq_len的分类，用于分类头尾实体是否有关系
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # [batch_size, num_type, seq_len, seq_len]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_type, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12

        return logits / self.config.hidden_size ** 0.5


class GlobalLinker(nn.Module):
    def __init__(self, config):
        super(GlobalLinker, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(os.environ["project_root"] + self.config.bert_path)
        self.BiLSTM = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768 // 2, batch_first=True)
        self.entity_model = GlobalPointer(self.config, 2)  # 2:头实体和尾实体
        self.head_category_model = GlobalPointer(self.config, self.config.num_category)  # 类别head position
        self.tail_category_model = GlobalPointer(self.config, self.config.num_category)  # 类别tail position
        self.head_sentiment_model = GlobalPointer(self.config, self.config.senti_polarity)  # 情感head position, senti_polarity:2
        self.tail_sentiment_model = GlobalPointer(self.config, self.config.senti_polarity)  # 情感tail position, senti_polarity:2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        hidden_state = self.bert(input_ids, attention_mask=attention_mask)
        lstm_out, (hidden_last, cn_last) = self.BiLSTM(hidden_state[0])
        entity_logits = self.entity_model(lstm_out, attention_mask)
        head_relation_logits = self.head_category_model(lstm_out, attention_mask)
        tail_relation_logits = self.tail_category_model(lstm_out, attention_mask)
        head_sentiment_logits = self.head_sentiment_model(lstm_out, attention_mask)
        tail_sentiment_logits = self.tail_sentiment_model(lstm_out, attention_mask)
        return entity_logits, head_relation_logits, tail_relation_logits, head_sentiment_logits, tail_sentiment_logits
