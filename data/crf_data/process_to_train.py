import os
import json
from tqdm import trange
from sklearn.model_selection import train_test_split, KFold


def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f"{desc}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_data_to_json(base_dir, save_data=False, save_dict=False):
    stack_examples = []
    pseudo_examples = []
    test_examples = []

    stack_dir = os.path.join(base_dir, "stack.json")
    # pseudo_dir = os.path.join(base_dir, "pseudo")
    # test_dir = os.path.join(base_dir, "test")

    # process train examples
    with open(stack_dir, "r") as f:
        stack_examples = json.load(f)

    # 构建实体知识库
    kf = KFold(10)
    entities = set()
    ent_types = set()
    for _now_id, _candidate_id in kf.split(stack_examples):
        now = [stack_examples[_id] for _id in _now_id]
        candidate = [stack_examples[_id] for _id in _candidate_id]
        now_entities = set()

        for _ex in now:
            for _label in _ex["labels"]:
                ent_types.add(_label[1])

                if len(_label[-1]) > 1:
                    now_entities.add(_label[-1])
                    entities.add(_label[-1])
        # print(len(now_entities))
        for _ex in candidate:
            text = _ex["text"]
            candidate_entities = []

            for _ent in now_entities:
                if _ent in text:
                    candidate_entities.append(_ent)

            _ex["candidate_entities"] = candidate_entities
    # assert len(ent_types) == 13

    # process test examples
    # todo 这里需要改正
    # for i in trange(1000, 1500):
    #     with open(os.path.join(test_dir, f"{i}.txt"), encoding="utf-8") as f:
    #         text = f.read()

    #     candidate_entities = []
    #     for _ent in entities:
    #         if _ent in text:
    #             candidate_entities.append(_ent)

    #     test_examples.append(
    #         {"id": i, "text": text, "candidate_entities": candidate_entities}
    #     )

    entities = {"entities": list(entities)}
    save_info(base_dir, entities, "entities")

    train, dev = train_test_split(
        stack_examples, shuffle=True, random_state=222, test_size=0.2
    )

    if save_data:
        save_info(base_dir, stack_examples, "stack")
        save_info(base_dir, train, "train")
        save_info(base_dir, dev, "dev")
        # save_info(base_dir, test_examples, "test")

        # save_info(base_dir, pseudo_examples, "pseudo")

    if save_dict:
        ent_types = list(ent_types)
        span_ent2id = {_type: i + 1 for i, _type in enumerate(ent_types)}

        ent_types = ["O"] + [
            p + "-" + _type for p in ["B", "I", "E", "S"] for _type in list(ent_types)
        ]
        # print("maybe here is a bug", ent_types)
        crf_ent2id = {ent: i for i, ent in enumerate(ent_types)}

        mid_data_dir = os.path.join(base_dir, "mid_data")
        if not os.path.exists(mid_data_dir):
            os.mkdir(mid_data_dir)

        save_info(mid_data_dir, span_ent2id, "span_ent2id")
        save_info(mid_data_dir, crf_ent2id, "crf_ent2id")


def build_ent2query(data_dir):
    # 利用比赛实体类型简介来描述 query
    ent2query = {
        # 药物
        "Tloc": "找出肿瘤位置：指肿瘤所在的部位。",
        # 药物成分
        "This": "找出肿瘤组织学类型：指肝细胞癌的组织排列方式。",
        # 疾病
        "Tdiff": "找出分化程度：肿瘤的分化是指肿瘤组织在形态和功能上与某种正常组织的相似之处，相似的程度称为肿瘤的分化程度。",
        # 症状
        "Tnum": "找出肿瘤数量：指肿瘤的数目。",
        # 症候
        "Tsize": "找出肿瘤大小：指在显微镜下于内皮细胞衬覆的脉管腔内见到癌细胞巢团，以门静脉分支为主（含包膜内血管）。根据MVI的数量和分布情况进行风险分级。",
        # 疾病分组
        "Sate": "找出卫星子灶：指主瘤周边近癌旁肝组织内出现的肉眼或显微镜下小癌灶。",
        # 食物
        "LC": "找出肝硬化程度：各种病因引起的肝脏疾病的终末期病变，病变以慢性进行性、弥漫性的肝细胞变性坏死、肝内纤维组织增生和肝细胞结节状再生为基本病理特征，广泛增生的纤维组织分割原来的肝小叶并包绕成大小不等的假小叶，引起肝小叶结构及血管的破坏和改建。",
        # 食物分组
        "TNM": "找出病理分期：是美国癌症联合委员会和国际抗癌联盟建立的恶性肿瘤分期系统。T是指原发肿瘤、N为淋巴结、M为远处转移。",
        # 人群
        "Caps": "找出包膜：指包绕在肿瘤组织外层的纤维组织。",
        "MVI": "找出微血管癌栓：指在显微镜下于内皮细胞衬覆的脉管腔内见到癌细胞巢团，以门静脉分支为主（含包膜内血管）。根据MVI的数量和分布情况进行风险分级。",
    }

    with open(
        os.path.join(data_dir, "mid_data", "mrc_ent2id.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(ent2query, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    convert_data_to_json(
        "/home/xiaojin/Code/DeepNER/data/crf_data", save_data=True, save_dict=True
    )
    build_ent2query("/home/xiaojin/Code/DeepNER/data/crf_data")
