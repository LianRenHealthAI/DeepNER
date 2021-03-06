#%%
import os
from transformers import BertTokenizer
import json
from submit_fix import fix_blank_index, zip_file
import pickle

# from transformers.tokenization_bert import BertTokenizer

from src.utils.model_utils import (
    CRFModel,
    SpanModel,
    EnsembleCRFModel,
    EnsembleSpanModel,
)

from src.utils.functions_utils import load_model_and_parallel
from collections import defaultdict
import torch
from src.utils.evaluator import crf_decode, span_decode
import zipfile
from config import Config

args = Config()
args.load_config_from_yaml(last=True)

args.version = "single"

SUBMIT_DIR = args.submit_dir
VERSION = (
    args.version
)  # choose single or ensemble or mixed ; if mixed  VOTE and TAST_TYPE is useless.
MID_DATA_DIR = args.mid_data_dir
TEST_DATA = "data/crf_data/test_data/test.json"

# BERT_DIR = "pretrained/chinese-roberta-wwm-ext"
BERT_DIR = args.bert_dir
print("bert dir", BERT_DIR)
# BERT_DIR = "pretrained/torch_uer_large"

# TASK_TYPE = "crf"
# GPU_IDS = "0"
# MAX_SEQ_LEN = 128

args.gpu_ids = "0"
VOTE = args.vote

LAMBDA = 0.3
THRESHOLD = 0.9

BERT_DIR_LIST = ["pretrained/bert-base-chinese"]


with open("./best_ckpt_path.txt", "r", encoding="utf-8") as f:
    CKPT_PATH = f.readlines()[-1].strip()

with open("./best_ckpt_path.txt", "r", encoding="utf-8") as f:
    ENSEMBLE_DIR_LIST = f.readlines()
    # print("ENSEMBLE_DIR_LIST:{}".format(ENSEMBLE_DIR_LIST))


def prepare_info():
    info_dict = {}
    with open(os.path.join(MID_DATA_DIR, f"crf_ent2id.json"), encoding="utf-8") as f:
        ent2id = json.load(f)

    with open(os.path.join(TEST_DATA), encoding="utf-8") as f:
        info_dict["examples"] = json.load(f)

    info_dict["id2ent"] = {ent2id[key]: key for key in ent2id.keys()}

    # info_dict["tokenizer"] = BertTokenizer.from_pretrained(BERT_DIR)
    info_dict["tokenizer"] = BertTokenizer(os.path.join(BERT_DIR, "vocab.txt"))

    return info_dict


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [" ", "\t", "\n"]:
            # todo 确定token作用
            tokens.append("[BLANK]")  # to do
            # tokens.append(_ch)
            # tokens.append("的")
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append("[INV]")
            else:
                tokens.append(_ch)

    return tokens


def base_predict(model, device, info_dict, ensemble=False, mixed=""):
    labels = defaultdict(list)

    tokenizer = info_dict["tokenizer"]
    id2ent = info_dict["id2ent"]

    with torch.no_grad():
        for _ex in info_dict["examples"]:
            ex_idx = _ex["id"]
            raw_text = _ex["text"]

            if not len(raw_text):
                labels[ex_idx] = []
                print("{}为空".format(ex_idx))
                continue

            sentences = raw_text
            # sentences = raw_text
            start_index = 0

            for sent in sentences:

                sent_tokens = fine_grade_tokenize(sent, tokenizer)
                # sent_tokens = list(sent)

                encode_dict = tokenizer.encode_plus(
                    text=sent_tokens,
                    max_length=args.max_seq_len,
                    # is_pretokenized=True,
                    # pad_to_max_length=True,
                    padding="max_length",  # todo 这啥为啥是false
                    is_split_into_words=True,
                    return_tensors="pt",
                    return_token_type_ids=True,
                    return_attention_mask=True,
                )

                model_inputs = {
                    "token_ids": encode_dict["input_ids"],
                    "attention_masks": encode_dict["attention_mask"],
                    "token_type_ids": encode_dict["token_type_ids"],
                }

                for key in model_inputs:
                    model_inputs[key] = model_inputs[key].to(device)

                if ensemble:
                    if args.task_type == "crf":
                        if VOTE:
                            decode_entities = model.vote_entities(
                                model_inputs, sent, id2ent, THRESHOLD
                            )
                        else:
                            pred_tokens = model.predict(model_inputs)[0]
                            decode_entities = crf_decode(pred_tokens, sent, id2ent)
                    else:
                        if VOTE:
                            decode_entities = model.vote_entities(
                                model_inputs, sent, id2ent, THRESHOLD
                            )
                        else:
                            start_logits, end_logits = model.predict(model_inputs)
                            start_logits = (
                                start_logits[0].cpu().numpy()[1 : 1 + len(sent)]
                            )
                            end_logits = end_logits[0].cpu().numpy()[1 : 1 + len(sent)]

                            decode_entities = span_decode(
                                start_logits, end_logits, sent, id2ent
                            )

                else:

                    if mixed:
                        if mixed == "crf":
                            pred_tokens = model(**model_inputs)[0][0]
                            decode_entities = crf_decode(pred_tokens, sent, id2ent)
                        else:
                            start_logits, end_logits = model(**model_inputs)

                            start_logits = (
                                start_logits[0].cpu().numpy()[1 : 1 + len(sent)]
                            )
                            end_logits = end_logits[0].cpu().numpy()[1 : 1 + len(sent)]

                            decode_entities = span_decode(
                                start_logits, end_logits, sent, id2ent
                            )

                    else:
                        if args.task_type == "crf":
                            pred_tokens = model(**model_inputs)[0][0]
                            decode_entities = crf_decode(pred_tokens, sent, id2ent)
                        else:
                            start_logits, end_logits = model(**model_inputs)

                            start_logits = (
                                start_logits[0].cpu().numpy()[1 : 1 + len(sent)]
                            )
                            end_logits = end_logits[0].cpu().numpy()[1 : 1 + len(sent)]

                            decode_entities = span_decode(
                                start_logits, end_logits, sent, id2ent
                            )

                for _ent_type in decode_entities:
                    for _ent in decode_entities[_ent_type]:
                        tmp_start = _ent[1] + start_index
                        tmp_end = tmp_start + len(_ent[0])

                        # try:
                        #     assert sent[tmp_start:tmp_end] == _ent[0]
                        # except:

                        #     print("-----")
                        #     print("sent[tmp_start:tmp_end]:", sent[tmp_start:tmp_end])
                        #     print("_ent[0]", _ent[0])
                        #     print(sent)
                        #     exit(1)

                        labels[ex_idx].append((_ent_type, tmp_start, tmp_end, _ent[0]))

                start_index += len(sent)

                if not len(labels[ex_idx]):
                    labels[ex_idx] = []

    return labels


#%%
def single_predict():
    save_dir = os.path.join(SUBMIT_DIR, VERSION)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    info_dict = prepare_info()

    print("info_dict id2ent", info_dict["id2ent"])
    print("len", len(info_dict["id2ent"]))

    model = CRFModel(bert_dir=BERT_DIR, num_tags=len(info_dict["id2ent"]))

    print(f"Load model from {CKPT_PATH}")
    model, device = load_model_and_parallel(model, args.gpu_ids, CKPT_PATH)
    # print(model)
    model.eval()

    labels = base_predict(model, device, info_dict)
    return labels


def ensemble_predict():
    save_dir = os.path.join(SUBMIT_DIR, VERSION)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    info_dict = prepare_info()

    model_path_list = [x.strip() for x in ENSEMBLE_DIR_LIST]
    print("model_path_list:{}".format(model_path_list))
    print('info_dict["id2ent"]')

    device = torch.device(f"cuda:{args.gpu_ids[0]}")

    if args.task_type == "crf":
        model = EnsembleCRFModel(
            model_path_list=model_path_list,
            bert_dir_list=BERT_DIR_LIST,
            num_tags=len(info_dict["id2ent"]),
            device=device,
            lamb=LAMBDA,
        )
    else:
        model = EnsembleSpanModel(
            model_path_list=model_path_list,
            bert_dir_list=BERT_DIR_LIST,
            num_tags=len(info_dict["id2ent"]) + 1,
            device=device,
        )

    labels = base_predict(model, device, info_dict, ensemble=True)

    return labels


#%%
def zip_file(src_dir):
    zip_name = "results.zip"
    z = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # print(dirpath, dirnames, filenames)
        if dirpath == "submit/results":
            fpath = dirpath.replace(src_dir, "")
            fpath = fpath and fpath + os.sep or ""
            for filename in filenames:
                z.write(os.path.join(dirpath, filename), fpath + filename)
    print("==压缩成功==")
    z.close()


def write_to_submit(labels):
    save_dir = os.path.join(SUBMIT_DIR, VERSION)
    print("save_dir: ", save_dir)
    for key in labels.keys():
        print(labels)
        with open(os.path.join(save_dir, f"{key}.tag"), "w", encoding="utf-8") as f:
            if not len(labels[key]):
                print(key)
                f.write("")
            else:
                for idx, _label in enumerate(labels[key]):
                    # if idx == len(labels[key]):
                    #     f.write(f"{_label[1]}#{_label[2]}#{_label[0]}#{_label[3]}")
                    # else:
                    f.write(f"{_label[1]}#{_label[2]}#{_label[0]}#{_label[3]}\n")


if __name__ == "__main__":

    # args = Config()
    # args.load_config_from_yaml(last=True)
    # args.gpu_ids = "0"
    # args.version = "single"
    # args.version = "ensemble"

    # if args.version == "single":
    #     print("========single predict=========")
    #     labels = single_predict()
    # else:
    #     print("========ensemble predict=========")
    #     labels = ensemble_predict()

    # with open("label.pkl", "wb") as f:
    #     pickle.dump(labels, f)

    with open("label.pkl", "rb") as f:
        labels = pickle.load(f)

    write_to_submit(labels)
    print("文件写入成功")

    fix_blank_index(
        "data/crf_data/test_data",
        "submit",
        version=args.version,
    )
    zip_file("submit")
