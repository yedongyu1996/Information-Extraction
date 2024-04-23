
import json
gold_num = {"dev": 0, "test": 0, "train": 0}
for path in ["train", "dev", "test"]:
    with open(path + "_convert.json", "r+", encoding="utf-8")as f:
        data = json.load(f)
    max = 0
    for da in data:
        if len(da["words"]) > max:
            max = len(da["words"])
    print(max)

