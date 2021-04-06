import json
import os

test_data_dir = "/home/xiaojin/Code/DeepNER/data/crf_data/test_data"


def convert_to_json(data_dir):
    file_name_list = [i for i in os.listdir(data_dir) if i.endswith("txt")]
    results = []

    for file_name in file_name_list:
        print(file_name)
        with open(os.path.join(data_dir, file_name), encoding="gbk") as f:
            results.append({"id": file_name.split(".")[0], "text": f.readlines()})

    with open(os.path.join(test_data_dir, f"test.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    convert_to_json(test_data_dir)
