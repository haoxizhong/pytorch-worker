import json
import torch
import numpy as np
import os

from formatter.Basic import BasicFormatter
from pytorch_pretrained_bert.tokenization import BertTokenizer

class BertCailFormatter(BasicFormatter):
    labelToId = {}
    idToLabel = {}

    def __init__(self, config, mode, *args, **params):
        super().__init__(self, config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.get("model", "bert_path"), 'vocab.txt'))
        self.mode = mode

        self.max_len = config.getint("data", "max_len")
        min_freq = config.getint("data", "min_freq")
        
        
        self.crit_label = {}
        with open(config.get("data", "crit_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                label = arr[0].replace("[", "").replace("]", "")
                cnt = int(arr[1])
                if cnt >= min_freq:
                    self.crit_label[label] = len(self.crit_label)

        
        self.law_label = {}
        with open(config.get("data", "law_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                x1 = int(arr[0])
                x2 = int(arr[1])
                cnt = int(arr[2])
                label = (x1, x2)
                if cnt >= min_freq:
                    self.law_label[label] = len(self.law_label)

        config.set('data', 'law_outdim', len(self.law_label))
        config.set('data', 'charge_outdim', len(self.crit_label))

        print("%d %d" % (len(self.crit_label), len(self.law_label)))
        

        f = open(config.get('data', 'attribute_path'), 'r')
        self.attr = json.loads(f.read())
        self.attr_vec = [[1, 0], [0, 1], [0, 0]]

        
        '''
        self.word2id = {}
        with open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r") as f:
            for line in f:
                self.word2id[line[:-1]] = len(self.word2id)
        '''
        


    def check_crit(self, data):
        cnt = 0
        for x in data:
            if x in self.crit_label.keys():
                cnt += 1
            else:
                return False
        return cnt == 1

    def check_law(self, data):
        arr = []
        for x, y, z in data:
            if x < 102 or x > 452:
                continue
            if not ((x, y) in self.law_label.keys()):
                return False
            arr.append((x, y))
            
        arr = list(set(arr))
        arr.sort()
        
        cnt = 0
        for x in arr:
            if x in arr:
                cnt += 1  # return False
        
        return cnt == 1

    def getAttribute(self, charge):
        try:
            attr = self.attr[charge]
        except:
            # print('gg?', charge)
            attr = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        
        return [self.attr_vec[v] for v in attr]



    def check(self, data, config):
        data = json.loads(data)
        if len(data["meta"]["criminals"]) != 1:
            return None
        if len(data["meta"]["crit"]) == 0 or len(data["meta"]["law"]) == 0:
            return None
        if not (self.check_crit(data["meta"]["crit"])):
            return None
        if not (self.check_law(data["meta"]["law"])):
            return None
         
        return data

    def get_crit_id(self, data):
        for x in data:
            if x in self.crit_label.keys():
                return self.crit_label[x], self.getAttribute(x)

    def get_law_id(self, data):
        for x in data:
            y = (x[0], x[1])
            if y in self.law_label.keys():
                return self.law_label[y]

    def get_time_id(self, data):
        v = 0
        if len(data["youqi"]) > 0:
            v1 = data["youqi"][-1]
        else:
            v1 = 0
        if len(data["guanzhi"]) > 0:
            v2 = data["guanzhi"][-1]
        else:
            v2 = 0
            
        if len(data["juyi"]) > 0:
            v3 = data["juyi"][-1]
        else:
            v3 = 0
        v = max(v1, v2, v3)

        if data["sixing"]:
            opt = 0
        elif data["wuqi"]:
            opt = 0
        elif v > 10 * 12:
            opt = 1
        elif v > 7 * 12:
            opt = 2
        elif v > 5 * 12:
            opt = 3
        elif v > 3 * 12:
            opt = 4
        elif v > 2 * 12:
            opt = 5
        elif v > 1 * 12:
            opt = 6
        elif v > 9:
            opt = 7
        elif v > 6:
            opt = 8
        elif v > 0:
            opt = 9
        else:
            opt = 10
        
        return opt


    def lookup(self, text, max_len):
        token = self.tokenizer.tokenize(text)
        token = ["[CLS]"] + token

        while len(token) < max_len:
            token.append("[PAD]")
        token = token[0:max_len]
        
        token = self.tokenizer.convert_tokens_to_ids(token)
        return token

    
    def process(self, data, config, mode, *args, **params):
        label = {'law': [], 'charge': [], 'time': [], 'attribute': []}
        passage = []

        for x in data:
            text = ''# .join(x['content'])
            for y in x['content']:
                for z in y:
                    text += z
            passage.append(self.lookup(text, self.max_len))

            label['law'].append(self.get_law_id(x['meta']['law']))
            charge, attr = self.get_crit_id(x['meta']['crit'])
            label['time'].append(self.get_time_id(x['meta']['time']))
            label['charge'].append(charge)
            label['attribute'].append(attr)

        for key in label:
            label[key] = torch.tensor(label[key], dtype = torch.long)
        passage = torch.tensor(passage, dtype = torch.long)

        return {'docs': passage, 'label_law': label['law'], 'label_charge': label['charge'],
                                'label_time': label['time'], 'label_attr': label['attribute']}
            


