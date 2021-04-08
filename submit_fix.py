# 更改换行带来的索引漂移问题
import os
import zipfile


def fix_blank_index(raw_txt_dir, tag_dir, version):
    raw_files = [f for f in os.listdir(raw_txt_dir) if f.endswith("txt")]

    # print(raw_files)
    # tag_files = os.listdir(tag_dir)
    for file_name in raw_files:
        with open(os.path.join(raw_txt_dir, file_name), "r", encoding="gbk") as f:
            print(file_name)
            raw_text = f.read()

            # 找出换行符的位置索引
            blank_idx = []
            for idx, char in enumerate(raw_text):
                if char == "\n":
                    blank_idx.append(1)
                else:
                    blank_idx.append(0)

            fix_results = []
            with open(
                os.path.join(tag_dir, version, file_name.replace("txt", "tag")),
                "r",
                encoding="utf8",
            ) as ft:

                for line in ft:
                    label_info = line.split("#")
                    # print("label_info:", label_info)
                    start_idx, end_idx = int(label_info[0]), int(label_info[1])
                    # 看前面有多少空字符了，直接加

                    pre_blank_num = sum(blank_idx[:start_idx])
                    label_info[0] = int(label_info[0]) + pre_blank_num
                    label_info[1] = int(label_info[1]) + pre_blank_num
                    fix_results.append(label_info)
            with open(
                os.path.join("submit", "results", file_name.replace("txt", "tag")), "w"
            ) as fr:
                for i in fix_results:
                    # 这里，解决嵌套
                    label = i[2]
                    entity = i[3]
                    if label == "MVI_Sate":
                        label_row_1 = [str(j) for j in i]
                        label_row_1[2] = "MVI"
                        fr.write("#".join(label_row_1))

                        label_row_2 = label_row_1.copy()
                        label_row_2[2] = "Sate"
                        fr.write("#".join(label_row_2))
                    else:
                        if entity == "脂肪性肝炎样":
                            continue
                        fr.write("#".join([str(j) for j in i]))


def zip_file(src_dir):
    zip_name = "results.zip"
    z = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir, "")
        fpath = fpath and fpath + os.sep or ""
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
    print("==压缩成功==")
    z.close()


if __name__ == "__main__":
    print("------")
    fix_blank_index("data/crf_data/test_data", "results/mixed")
    zip_file("submit")
