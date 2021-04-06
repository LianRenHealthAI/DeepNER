#%%
import jsonlines
import json
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
# doccano_file_path = "/home/xiaojin/Code/DeepNER/data/crf_data/ann-0405.jsonl"
# doccano_file_path = "/home/xiaojin/Code/DeepNER/data/crf_data/file.json1"
doccano_file_path = "/home/xiaojin/Code/DeepNER/data/crf_data/train-ann-0325.json1"
official_file_path = "/home/xiaojin/Code/DeepNER/data/crf_data/file_official.json1"
write_path = "/home/xiaojin/Code/DeepNER/data/crf_data/"

#%%
def convert_to_df(doccano_file_path, official=False, refine_file=False):
    with jsonlines.open(doccano_file_path) as f:
        """
        {"id": 1, "text": "EU rejects ...",
        "labels": [[0,2,"ORG"], [11,17, "MISC"], [34,41,"ORG"]]}
        """
        labeled_data = []
        for line in f:
            # print(line)
            text = line["text"]
            if official:
                labels = [[l[0], l[1], l[2]] for l in line["labels"]]
            elif refine_file:
                annotations = line["annotations"]
                label_maping = {
                    1: "Caps",
                    2: "LC",
                    3: "MVI",
                    4: "Sate",
                    5: "TNM",
                    6: "Tdiff",
                    7: "This",
                    8: "Tloc",
                    9: "Tnum",
                    10: "Tsize",
                    11: "Caps",
                    12: "LC",
                    13: "MVI",
                    14: "Sate",
                    15: "TNM",
                    16: "Tdiff",
                    17: "This",
                    18: "Tloc",
                    19: "Tnum",
                    20: "Tsize",
                }
                labels = []
                if annotations:
                    for i in annotations:
                        start_idx = i["start_offset"]
                        end_idx = i["end_offset"]
                        label = label_maping[i["label"]]
                        labels.append([start_idx, end_idx, label])
            else:
                text = line["text"]
                labels = []
                if line["labels"]:
                    labels = [[l[0], l[1], l[2]] for l in line["labels"]]

            tokens = list(text)

            labeled_data_dict = {}
            for idx, char in enumerate(tokens):
                labeled_data_dict[idx] = [char, "O"]

            for label_ in labels:
                start_idx, end_idx, label = label_
                for i in range(start_idx, end_idx):
                    if i == start_idx:
                        labeled_data_dict[i][1] = f"B-{label}"
                    else:
                        labeled_data_dict[i][1] = f"I-{label}"
            labeled_data += labeled_data_dict.values()

    # 分段
    labeled_data_list, labeled_data_seq = [], []

    for data in labeled_data:
        # if data[0] == "\n":
        if data[0] == "。":
            if labeled_data_seq:
                labeled_data_list.append(labeled_data_seq)
            labeled_data_seq = []
        elif data[0] == " ":
            continue
        else:
            labeled_data_seq.append(data)

    # space

    return labeled_data_list


def convert_to_format(labeled_data):
    saved_data_list = []
    for n, seq in enumerate(labeled_data):
        text = "".join([r[0] for r in seq])
        labels = []
        # entity = ""
        # pre_char_slash = False
        start_idx = None
        end_idx = None
        num_entity = 0

        for idx, lab_ in enumerate(seq):
            _, lab = lab_
            if "-" in lab:
                if not start_idx:
                    start_idx = idx
                end_idx = idx
                label = lab.split("-")[1]
            else:
                if end_idx:
                    num_entity += 1
                    entity = text[start_idx : end_idx + 1]
                    span = (
                        f"T{num_entity}",
                        label,
                        start_idx,
                        end_idx,
                        entity,
                    )
                    labels.append(span)

                    assert len(entity) == end_idx + 1 - start_idx
                    end_idx = None
                    start_idx = None

        saved_data = {
            "id": n,
            "text": text,
            "labels": labels,
            "pseudo": 0,  # 伪标签
            "candidate_entities": [],  # 远程监督
        }
        saved_data_list.append(saved_data)
    return saved_data_list


#%%
data_mid_1 = convert_to_df(doccano_file_path)
# data_mid_1 = convert_to_df(doccano_file_path)
data_mid_2 = convert_to_df(official_file_path, official=True)
data_mid = data_mid_1 + data_mid_2

data_mid = data_mid_1
print(data_mid[0])

labels = []
for s in data_mid:
    for _, label in s:
        labels.append(label)
labels = list(set(labels))
labels_dict = {l: i for i, l in enumerate(labels)}
with open(write_path + "crf_ent2id.json", "w+") as f:
    json.dump(labels_dict, f, ensure_ascii=False, indent=2)


data = convert_to_format(data_mid)

with open(write_path + "/stack.json", "w+") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# train, dev = train_test_split(data, test_size=0.2)

# with open(write_path + "/train.json", "w+") as f:
#     json.dump(train, f, ensure_ascii=False, indent=2)

# with open(write_path + "/dev.json", "w+") as f:
#     json.dump(dev, f, ensure_ascii=False, indent=2)

# %%
