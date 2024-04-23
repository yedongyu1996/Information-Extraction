
import json
gold_num = {"dev": 0, "test": 0, "train": 0}
for path in ["train", "dev", "test"]:
    with open(path + "_convert.json", "r+", encoding="utf-8")as f:
        data = json.load(f)
    tmp_data = []
    for ins in data:
        gold_num[path] += len(ins["triple_list"])

print(gold_num)

