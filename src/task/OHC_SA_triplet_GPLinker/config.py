import json
import os
try:
    os.environ['project_root']
except KeyError:
    os.environ["project_root"] = os.path.abspath("../../../")
class Config(object):
    def __init__(self):
        # self.dataset = "bart_format"
        self.dataset = "OHC_SA_triplet"
        self.train_data = "data/{}/train_convert.json".format(self.dataset)
        self.dev_data = "data/{}/dev_convert.json".format(self.dataset)
        self.test_data = "data/{}/test_convert.json".format(self.dataset)
        self.map_rel = "data/{}/med_rel2id.json".format(self.dataset)
        self.model_name = "GPLinker"
        self.batch_size = 4
        self.hidden_size = 64
        self.learning_rate = 2e-5
        self.bert_path = "/resource/bert-base-chinese"

        self.dev_result = "/src/task/OHC_SA_triplet_GPLinker/evaluate_result/dev.json"
        self.epochs = 600
        self.RoPE = True
        self.bert_dim = 768

        with open(os.environ["project_root"] + "/" + self.map_rel, "r", encoding="utf-8") as f:
            self.num_rel = len(json.load(f)[0])
