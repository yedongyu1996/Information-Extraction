from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter


class TripletSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, conflict_id, opinion_first=True):
        super(TripletSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.word_start_index = num_labels + 2

        self.FN = 0
        self.FP = 0
        self.TP = 0

        self.target_num = 0
        self.pred_num = 0
        self.corr_num = 0

        self.em = 0
        self.total = 0
        self.invalid = 0
        self.conflict_id = conflict_id
        # assert opinion_first is False, "Current metric only supports aspect first"

    def evaluate(self, entity_relation_entity_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉[CLS]
        tgt_tokens = tgt_tokens[:, 1:]  # 去掉[CLS]

        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        """
                size：batch
                每个元素存储的是每个pred的真实有效长度，
                即除去CLS和SEP以及PAD后的长度。 PAD是SEP 。长度包括了任务类型
        """
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        """
                size：batch
                每个元素存储的是每个pred的真实有效长度，
                即除去CLS和SEP以及PAD后的长度。 PAD是SEP 。长度包括了任务类型
        """
        target_seq_len = (target_seq_len - 2).tolist()

        pred_spans = []
        for i, (ts, ps) in enumerate(zip(entity_relation_entity_span, pred.tolist())):
            em = 0
            assert ps[0] == tgt_tokens[i, 0]
            ps = ps[2:pred_seq_len[i]]  # 预测值，除去任务符号的有效长度片段

            # 当预测序列长度和真实序列长度相时，看看有多少个是完全预测出来了
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):  # 循环里面是正确找出一对对的asp或者opin对（from，to）
                for index, j in enumerate(ps):
                    if j < self.word_start_index:  # 预测的可能是AE_OE_SC 或者 POS NEG这些
                        cur_pair.append(j)
                        """
                        1.len(cur_pair)!=5 如果抽取到了pos或者neg，但是抽取的一对长度小于5
                        2.cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]，方面词和评论词头尾有颠倒
                        """
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))  # 如果抽取了一对的话，添加进cur_pair
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())
            self.invalid += invalid

            target_counter = Counter()
            pred_counter = Counter()

            for t in ts:
                target_counter[(t[0], t[1], t[2], t[3], t[4])] = 1

            for p in pairs:
                pred_counter[(p[0], p[1], p[2], p[3], p[4])] = 1

            tp, fn, fp = _compute_tp_fn_fp(
                [(key[0], key[1], key[2], key[3], key[4], value) for key, value in pred_counter.items()],
                [(key[0], key[1], key[2], key[3], key[4], value) for key, value in target_counter.items()]
            )
            self.FN += fn
            self.FP += fp
            self.TP += tp
            self.target_num += len(target_counter)
            self.pred_num += len(pred_counter)
            self.corr_num += tp

            # sorry, this is a very wrongdoing, but to make it comparable with previous work, we have to stick to the
            #   error
            # for key in aeosc_pred_counter:
            #     if key not in aeosc_target_counter:
            #         continue
            #     if aeosc_target_counter[key]==aeosc_pred_counter[key]:
            #         self.sc_tp[aeosc_pred_counter[key]] += 1
            #         aeosc_target_counter.pop(key)
            #     else:
            #         self.sc_fp[aeosc_pred_counter[key]] += 1
            #         self.sc_fn[aeosc_target_counter[key]] += 1

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.TP, self.FN, self.FP)
        res['F1'] = round(f * 100, 2)
        res['R'] = round(rec * 100, 2)
        res['P'] = round(pre * 100, 2)
        res['gold_num'] = self.target_num
        res['pred_num'] = self.pred_num
        res['corr_num'] = self.corr_num


        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        if reset:
            self.FP = 0
            self.TP = 0
            self.FN = 0

        return res


class AspectOpinionExtractionSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, conflict_id, opinion_first=True):
        super(AspectOpinionExtractionSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.word_start_index = num_labels + 2

        self.FN = 0
        self.FP = 0
        self.TP = 0

        self.em = 0
        self.total = 0
        self.invalid = 0
        self.conflict_id = conflict_id

    def evaluate(self, aspect_opinion_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉[CLS]
        tgt_tokens = tgt_tokens[:, 1:]  # 去掉[CLS]

        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        """
                size：batch
                每个元素存储的是每个pred的真实有效长度，
                即除去CLS和SEP以及PAD后的长度。 PAD是SEP 。长度包括了任务类型
        """
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        """
                size：batch
                每个元素存储的是每个pred的真实有效长度，
                即除去CLS和SEP以及PAD后的长度。 PAD是SEP 。长度包括了任务类型
        """
        target_seq_len = (target_seq_len - 2).tolist()

        pred_spans = []
        for i, (ts, ps) in enumerate(zip(aspect_opinion_span, pred.tolist())):
            em = 0
            assert ps[0] == tgt_tokens[i, 0]
            ps = ps[2:pred_seq_len[i]]  # 预测值，除去任务符号的有效长度片段

            # 当预测序列长度和真实序列长度相时，看看有多少个是完全预测出来了
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            invalid = 0
            pairs = []
            cur_pair = []
            if len(ps):  # 循环里面是正确找出一对对的asp或者opin对（from，to）
                for index, j in enumerate(ps):
                    if j < self.word_start_index:  # 预测的可能是AE_OE_SC 或者 POS NEG这些
                        cur_pair.append(j)
                        """
                        1.len(cur_pair)!=5 如果抽取到了pos或者neg，但是抽取的一对长度小于5
                        2.cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]，方面词和评论词头尾有颠倒
                        """
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                            invalid = 1
                        else:
                            pairs.append(tuple(cur_pair))  # 如果抽取了一对的话，添加进cur_pair
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())
            self.invalid += invalid

            aspect_opinion_target_counter = Counter()
            aspect_opinion_pred_counter = Counter()

            for t in ts:
                aspect_opinion_target_counter[(t[0], t[1])] = 1

            for p in pairs:
                aspect_opinion_pred_counter[(p[0], p[1])] = 1

            # 这里相同的pair会被计算多次
            tp, fn, fp = _compute_tp_fn_fp(
                [(key[0], key[1], value) for key, value in aspect_opinion_pred_counter.items()],
                [(key[0], key[1], value) for key, value in aspect_opinion_target_counter.items()]
            )
            self.FN += fn
            self.FP += fp
            self.TP += tp

            # sorry, this is a very wrongdoing, but to make it comparable with previous work, we have to stick to the
            #   error
            # for key in aeosc_pred_counter:
            #     if key not in aeosc_target_counter:
            #         continue
            #     if aeosc_target_counter[key]==aeosc_pred_counter[key]:
            #         self.sc_tp[aeosc_pred_counter[key]] += 1
            #         aeosc_target_counter.pop(key)
            #     else:
            #         self.sc_fp[aeosc_pred_counter[key]] += 1
            #         self.sc_fn[aeosc_target_counter[key]] += 1

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.TP, self.FN, self.FP)
        res['F1'] = round(f * 100, 2)
        res['R'] = round(rec * 100, 2)
        res['P'] = round(pre * 100, 2)

        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        if reset:
            self.FP = 0
            self.TP = 0
            self.FN = 0

        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0  # 正样本被预测为正样本
    fp = 0  # 负样本被预测为正样本
    fn = 0  # 正样本被预测为负样本
    if isinstance(ts, (list, set)):
        ts = {key: 1 for key in list(ts)}
    if isinstance(ps, (list, set)):
        ps = {key: 1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]  # 预测成功样本
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp
