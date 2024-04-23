import json


def extract_aspect(path):
    """
    提取所有的文档中所有的方面词
    :return:
    """
    with open(path, "r+", encoding="utf-8") as f:
        data = json.load(f)
    tmp_aspects = []
    for sample in data:
        aspects = sample["aspects"]
        tmp_aspects.append([''.join(i["term"]) for i in aspects])
    tmp_list = sum(tmp_aspects, [])
    with open("aspect.txt", "a+", encoding="utf-8") as f:
        for aspect in tmp_list:
            f.write(aspect)
            f.write('\n')


def input_category():
    """
    对原数据中的方面进行归类
    :return:
    """
    with open("aspect_category_dic.txt", "r+", encoding="utf-8")as f1:
        aspect_category = eval(f1.read().replace('\n',''))
    for path in ['train', 'test', 'dev']:
        with open(path + "_convert.json", "r+", encoding="utf-8")as f2:
            datas = json.load(f2)
        for data in datas:  # data: dic
            aspects = data["aspects"]
            for aspect in aspects:
                term = ''.join(aspect['term'])
                if term in aspect_category:
                    aspect["category"] = aspect_category[term]
                else:
                    aspect["category"] = "其他"
        with open(path + "_convert_cate.json", "w+", encoding="utf-8")as f3:
            json.dump(datas, f3, ensure_ascii=False)


if __name__ == "__main__":

    input_category()


