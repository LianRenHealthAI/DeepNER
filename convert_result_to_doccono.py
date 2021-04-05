import pandas as pd
import os
import json

results = []
for file in range(0, 21):
    with open(f"data/crf_data/test_data/{file}.txt", "r", encoding="gbk") as f:
        text = f.read()
    with open(f"submit/results/{file}.tag", "r") as f:
        ann = f.read()

    pos = list(range(len(text)))
    new_lines = 0
    for i in pos:
        if text[i] == "\n":
            pos[i] = -1
            new_lines += 1
        else:
            pos[i] = i + new_lines
    pos_dic = {j: i for i, j in enumerate(pos) if j > 0}

    entities = []
    for ett in ann.split("\n"):
        if ett:
            paras = ett.split("#")
            if paras[0] == "-1" or paras[1] == "-1":
                print(file, ett)
                continue
            entities.append((pos_dic[int(paras[0])], pos_dic[int(paras[1])], paras[2]))
    results.append({"text": text, "labels": entities})

with open("test.jsonl", "w") as f:
    f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in results]))
