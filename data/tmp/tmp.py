import json

with open('train_convert.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
    dis_dict = {"正面": 0, "负面": 0}
    for i in data:
        aspects = [''.join(aspect["term"]) for aspect in i["aspects"]]
        sentiment = ["正面" if aspect["polarity"] == "POS" else "负面" for aspect in i["aspects"]]
        opinions = [''.join(opinion["term"]) for opinion in i["opinions"]]
        for aspect_, sentiment_, opinion_ in zip(aspects, sentiment, opinions):
            if sentiment_ == "正面":
                dis_dict["正面"] += 1
            else:
                dis_dict["负面"] += 1
with open("data_positive.txt", "w+", encoding="utf-8")as fp:
    with open("data_negative.txt", "w+", encoding="utf-8") as fn:
        fp.write("[")
        fp.write('\n')
        fn.write("[")
        fn.write('\n')
        for i in data:
            aspects = [''.join(aspect["term"]) for aspect in i["aspects"]]
            sentiment = ["正面" if aspect["polarity"]=="POS" else "负面" for aspect in i["aspects"]]
            opinions = [''.join(opinion["term"]) for opinion in i["opinions"]]
            for aspect_, sentiment_, opinion_ in zip(aspects, sentiment, opinions):
                tmp_dict = {}
                if sentiment_=="正面":
                    tmp_dict["aspect"] = aspect_
                    tmp_dict["opinion"] = opinion_
                    tmp_dict["sentiment"] = sentiment_
                    fp.write("\t\t")
                    fp.write(str(tmp_dict))
                    fp.write(',')
                    fp.write('\n')
                else:
                    tmp_dict["aspect"] = aspect_
                    tmp_dict["opinion"] = opinion_
                    tmp_dict["sentiment"] = sentiment_
                    fn.write("\t\t")
                    fn.write(str(tmp_dict))
                    fn.write(',')
                    fn.write('\n')
        fp.write(']')
        fn.write(']')
        print(dis_dict)




if __name__ == "__main__":
    pass