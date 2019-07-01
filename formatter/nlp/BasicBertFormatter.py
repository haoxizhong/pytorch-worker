import json
import torch
import os

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter


class BasicBertFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        input = []
        if mode != "test":
            label = []

        for temp in data:
            text = temp["text"]
            token = self.tokenizer.tokenize(text)

            while len(token) < self.max_len:
                token.append("[PAD}")

            input.append(token)
            if mode != "test":
                label.append(temp["label"])

        input = torch.LongTensor(input)
        if mode != "test":
            label = torch.LongTensor(label)

        if mode != "test":
            return {'input': input, 'label': label}
        else:
            return {"input": input}
