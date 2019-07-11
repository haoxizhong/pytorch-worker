import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import json

# from utils.util import calc_accuracy, gen_result, generate_embedding
# from model.model.demo.TextCNN import TextCNN
from pytorch_pretrained_bert import BertModel
from tools.accuracy_init import init_accuracy_function




class MultiTaskBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MultiTaskBert, self).__init__()

        self.taskName = config.get('data', 'task_name').split(',')
        self.taskName = [v.strip() for v in self.taskName]

        # self.cnn = TextCNN(config)
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))


        min_freq = config.getint("data", "min_freq")
        self.crit_label = {}
        with open(config.get("data", "crit_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                label = arr[0].replace("[", "").replace("]", "")
                cnt = int(arr[1])
                if cnt >= min_freq:
                    self.crit_label[label] = len(label)
        
        self.law_label = {}
        with open(config.get("data", "law_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                x1 = int(arr[0])
                x2 = int(arr[1])
                cnt = int(arr[2])
                label = (x1, x2)
                if cnt >= min_freq:
                    self.law_label[label] = len(label)
        task_name_num = {
            'law': len(self.law_label),
            'charge': len(self.crit_label),
            'time': 11
        }

        self.out = [nn.Linear(768,task_name_num[name]) for name in self.taskName]

        self.out = nn.ModuleList(self.out)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)
    
    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids = device)
        self.out = nn.DataParallel(self.out, device_ids = device)


    def forward(self, data, config, gpu_list, acc_result={'law': None, 'charge': None, 'time': None}, mode = 'train'):
        passage = data['docs']  # batch, len
        labels = {}
        labels['law'] = data['label_law']
        labels['charge'] = data['label_charge']
        labels['time'] = data['label_time']
        labels['attribute'] = data['label_attr']

        if acc_result is None:
            acc_result = {'law': None, 'charge': None, 'time': None}

        # passage = self.embs(passage)

        # passage = self.cnn(passage)
        _, passage = self.bert(passage, output_all_encoded_layers = False)
        passage = passage.view(passage.size()[0], -1)


        task_result = {}
        for i in range(len(self.taskName)):
            task_result[self.taskName[i]] = self.out.module[i](passage)
            # task_result.append(self.out[i](vec))

        # loss = self.criterion(task_result, labels)
        loss = self.criterion(task_result['law'], labels['law']) + self.criterion(task_result['charge'], labels['charge']) + self.criterion(task_result['time'], labels['time'])
        
        
        acc_result['law'] = self.accuracy_function(task_result['law'], labels['law'], config, acc_result['law'])
        acc_result['time'] = self.accuracy_function(task_result['time'], labels['time'], config, acc_result['time'])
        acc_result['charge'] = self.accuracy_function(task_result['charge'], labels['charge'], config, acc_result['charge'])
        
        
        '''
        result = {
            'law': torch.max(task_result['law'], dim=1)[1].cpu().numpy(),
            'charge': torch.max(task_result['charge'], dim=1)[1].cpu().numpy(),
            'time': torch.max(task_result['time'], dim=1)[1].cpu().numpy()
        }
        '''
        return {'loss': loss, 'acc_result': acc_result}
        
        
        # return {"loss": loss, "accuracy": accu, "result": result, "x": task_result, "accuracy_result": acc_result}
