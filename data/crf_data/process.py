import jsonlines
import json
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
doccano_file_path = "/home/xiaojin/Code/Flat-Lattice-Transformer/crf_data/file.json1"
write_path = "/home/xiaojin/Code/DeepNER/data/crf_data/stack.json"

#%%
def convert_to_df(doccano_file_path):
    with jsonlines.open(doccano_file_path) as f:
        """
        {"id": 1, "text": "EU rejects ...",
        "labels": [[0,2,"ORG"], [11,17, "MISC"], [34,41,"ORG"]]}
        """
        labeled_data = []
        for line in f:
            # print(line)
            text = line["text"][4:]
            tokens = list(text)
            labels = [[l[0] - 4, l[1] - 4, l[2]] for l in line["labels"]]
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

    # 分句
    labeled_data_list, labeled_data_seq = [], []

    for data in labeled_data:
        if data[0] == "\n":
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
        entity = ""
        pre_char_slash = False
        start_idx = None
        num_entity = 0
        for idx, (char, lab) in enumerate(seq):
            if "-" in lab:
                label = lab.split("-")[1]
                entity += char
                pre_char_slash = True
                if not start_idx:
                    start_idx = idx
            else:
                if pre_char_slash:
                    num_entity += 1
                    labels.append([f"T{num_entity}", label, start_idx, idx - 1, entity])
                    #                 entity_type.append(label)
                    pre_char_slash = False
                    entity = ""
                    start_idx = None
        saved_data = {"id": n, "text": text, "labels": labels}
        saved_data_list.append(saved_data)
    return saved_data_list


data = convert_to_format(convert_to_df(doccano_file_path))
with open(write_path, "w+") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
