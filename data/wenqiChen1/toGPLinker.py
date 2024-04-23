
import json
gold_num = {"dev": 0, "test": 0, "train": 0}
for path in ["train", "dev", "test"]:
    with open(path + "_convert.json", "r+", encoding="utf-8")as f:
        data = json.load(f)
    tmp_data = []
    for ins in data:
        gold_num[path] += len(ins["aspects"])
        triple_list = []
        text = ins["raw_words"].replace(' ','')
        entities1 = ins["aspects"]
        entities2 = ins["opinions"]
        for entity1, entity2 in zip(entities1, entities2):
            triple_list.append([''.join(entity1["term"]), entity1["polarity"], ''.join(entity2["term"])])
        tmp_data.append({"text": text, "triple_list": triple_list})
    # with open("./GPLinker/" + path + "_convert.json", "w+", encoding="utf-8")as fw:
    #     json.dump(tmp_data, fw,ensure_ascii=False)

print(gold_num)

