import torch
import numpy as np
import random

def padding(x, max_len, start_point):
    xx = x[start_point:] + x[:start_point]
    assert len(xx) == len(x)
    out_data = []
    while len(out_data) < max_len:
        out_data = out_data + xx
    if len(out_data) > max_len:
        out_data = out_data[: max_len]
    return out_data


class Basic_formatter:
    def __init__(self, config):
        self.max_seq_len = config.getint("train", "max_seq_len")
        self.task = config.get("train", "task").split(",")  # 例如0,1,2表示训0，1，2三种疾病
        self.task_dict = {}
        self.use_feature = config.getboolean("data", "use_feature")
        self.random_start = config.getboolean("data", "random_start")
        for idx, task in enumerate(self.task):
            self.task_dict[task] = idx

    def process(self, data, mode="train"):
        ecg_data = []
        label = []
        file = []
        feature = []
        if self.random_start and mode == "train":
            start_point = random.randint(0, len(data[0]["data"][0]) - 1)
        else:
            start_point = 0
        for item in data:
            for idx, lead in enumerate(item["data"]):
                item["data"][idx] = padding(item["data"][idx], self.max_seq_len, start_point)
            ecg_data.append(item["data"])
            label.append([0] * len(self.task))
            for lab in item["label"]:
                if str(lab) in self.task_dict.keys():
                    label[-1][self.task_dict[str(lab)]] = 1
            file.append(item["file"])
            feature.append(item["feature"])
        ecg_data = torch.from_numpy(np.array(ecg_data)).float()
        feature = torch.from_numpy(np.array(feature)).float()
        label = torch.from_numpy(np.array(label)).long()
        return {"input": ecg_data, "label": label, "file": file, "feature": feature}
