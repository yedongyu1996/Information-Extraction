
class Config(object):
    def __init__(self):
        self.dataset = "bart_format"
        self.num_rel = 6
        self.batch_size = 4
        self.hidden_size = 64
        self.learning_rate = 2e-5
        self.bert_path = "/resource/bert-base-chinese"

        self.train_data = "data/{}/train_convert.json".format(self.dataset)
        self.dev_data = "data/{}/dev_convert.json".format(self.dataset)
        self.test_data = "data/{}/test_convert.json".format(self.dataset)
        self.map_rel = "data/{}/med_rel2id_tmp.json".format(self.dataset)

        self.dev_result = "/src/task/OHC_SA_triplet_GPLinker/evaluate_result/dev.json"
        self.epochs = 600
        self.RoPE = True
        self.bert_dim = 768

