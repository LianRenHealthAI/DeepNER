import json
import os
import time


class Config:
    def __init__(self) -> None:
        self.gpu_ids = "0"
        self.output_dir = "out"
        self.mid_data_dir = "data/crf_data/mid_data"
        self.raw_data_dir = "data/crf_data"
        self.mode = "train"
        self.task_type = "crf"
        self.bert_dir = "pretrained/bert-base-chinese"
        self.bert_type = "bert-base"
        self.train_epochs = 10
        self.swa_start = 5
        self.attack_train = ""
        self.train_batch_size = 64
        self.eval_batch_size = 64
        self.dropout_prob = 0.1
        self.max_seq_len = 128
        self.lr = 2e-5
        self.other_lr = 2e-3
        self.seed = 123
        self.weight_decay = 0.01
        self.loss_type = "ls_ce"  # focal | ce | ls_ce
        self.eval_model = True
        self.use_fp16 = False
        self.use_type_embed = False
        self.max_grad_norm = 1.0
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.version = "v0"
        self.submit_dir = "submit"
        self.ckpt_dir = ""
        self.training_time = ""

    def load_config_from_json(self, path="training_params.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self, k, v)

    # @classmethod
    def save_config(self, folder="params_history"):
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(
            folder, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".json"
        )

        with open(file_path, "w") as f:
            print(self.__dict__)
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    config = Config()
    config.save_config()
