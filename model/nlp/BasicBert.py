import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from tools.accuracy_init import init_accuracy_function


class BertEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

    def forward(self, x):
        out = self.bert(x)

        return out.last_hidden_state[:,-1,:]


class BasicBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BasicBert, self).__init__()

        self.output_dim = config.getint("model", "output_dim")
        self.bert = BertEncoder(config, gpu_list, *args, **params)
        self.fc = nn.Linear(self.bert.bert.config.hidden_size, self.output_dim)

        self.seq = nn.Sequential(self.bert, self.fc)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        self.seq = nn.DataParallel(self.seq, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['input']

        y = self.seq(x)

        if "label" in data.keys():
            label = data["label"]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {}
