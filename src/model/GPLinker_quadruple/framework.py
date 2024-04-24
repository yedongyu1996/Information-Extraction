import torch
from models import GlobalLinker
from tqdm import tqdm
import json
import os
import numpy as np
from loss import sparse_multilabel_categorical_crossentropy as SMCC_loss
import time


def count_params(model):
    param_count = 0
    for index, param in enumerate(model.parameters()):
        param_count += param.view(-1).size()[0]
    print('total number of parameters: %d\n\n' % param_count)


class Framework(object):
    def __init__(self, config):
        self.config = config
        with open(os.environ["project_root"] + "/" + self.config.map_cate, "r", encoding="utf-8") as f:
            self.cate_id2label = json.load(f)[1]
        with open(os.environ["project_root"] + "/" + self.config.map_senti, "r", encoding="utf-8") as f:
            self.senti_id2label = json.load(f)[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_dataloader, dev_dataloader, test_dataloader):

        now = time.localtime()
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", now)

        model = GlobalLinker(self.config).to(self.device)

        count_params(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        best_epoch = 0
        best_f1_score = 0
        global_step = 0
        global_loss = 0
        best_threshold = None
        p, r = 0, 0
        for epoch in range(self.config.epochs):
            for data in tqdm(train_dataloader):
                entity_logtis, head_cate_logits, tail_cate_logits, head_senti_logits, tail_senti_logits = model(data)
                optimizer.zero_grad()
                entity_loss = SMCC_loss(data["entity_list"].to(self.device), entity_logtis, True)
                head_cate_loss = SMCC_loss(data["head_cate_list"].to(self.device), head_cate_logits, True)
                tail_cate_loss = SMCC_loss(data["tail_cate_list"].to(self.device), tail_cate_logits, True)
                head_senti_loss = SMCC_loss(data["head_senti_list"].to(self.device), head_senti_logits, True)
                tail_senti_loss = SMCC_loss(data["tail_senti_list"].to(self.device), tail_senti_logits, True)

                loss = sum([entity_loss + head_cate_loss + tail_cate_loss + head_senti_loss + tail_senti_loss]) / 5
                loss.backward()
                optimizer.step()
                global_loss += loss.item()
                global_step += 1
            print("epoch {} global_step: {} global_loss: {:5.4f}".format(epoch, global_step, global_loss))
            global_loss = 0
            if epoch >= 0:
                for threshold in [-0.5]:
                    precision, recall, f1_score, predict = self.evaluate(model, dev_dataloader, threshold=threshold)
                    if best_f1_score < f1_score:
                        best_f1_score = f1_score
                        best_threshold = threshold
                        # json.dump(predict, open(os.environ["project_root"] + self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
                        p, r = precision, recall
                        best_epoch = epoch
                        print("save model......")
                        torch.save(model.state_dict(), "checkpoint/" + "GPLinker_triplet_" + current_time + '.pt')
        print(
            "best_epoch: {} precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f} threshold: {}".format(best_epoch, p, r,
                                                                                                       best_f1_score,
                                                                                                       best_threshold))

    def evaluate(self, model, dataloader, threshold=-0.5):

        model.eval()
        predict_num, gold_num, correct_num = 0, 0, 0
        predict = []

        def to_tuple(data):
            tuple_data = []
            for i in data:
                tuple_data.append(tuple(i))
            return tuple(tuple_data)

        with torch.no_grad():
            for data in dataloader:
                text = data["text"][0]
                logits = model(data)
                outputs = [o.cpu()[0] for o in logits]
                outputs[0][:, [0, -1]] -= np.inf  # 2 * max_length * 2
                outputs[0][:, :, [0, -1]] -= np.inf

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
                                category = self.cate_id2label[str(c)]
                                senti = self.senti_id2label[str(s)]
                                predict.append((sh, st, category, senti, oh, ot))

                quadruple = data["quadruple_list"][0]
                quadruple = set(to_tuple(quadruple))
                predict = set(predict)
                correct_num += len(quadruple & predict)
                predict_num += len(predict)
                gold_num += len(quadruple)
                lack = quadruple - predict
                new = predict - quadruple
                # predict.add({"text": text, "gold": list(quadruple), "predict": list(predict), "lack": list(lack),
                #              "new": list(new)})
            recall = correct_num / (gold_num + 1e-10)
            precision = correct_num / (predict_num + 1e-10)
            f1_score = 2 * recall * precision / (recall + precision + 1e-10)
            print("correct_num:{} predict_num: {} gold_num: {}".format(correct_num, predict_num, gold_num), end=' ')
            print("precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(precision, recall, f1_score))
        return precision, recall, f1_score, predict
