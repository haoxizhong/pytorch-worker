import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class JsonFromMemDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding


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
        print("%d %d" % (len(self.crit_label), len(self.law_label)))



        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive")

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), recursive)
        self.file_list.sort()

        self.load_mem = config.getboolean("data", "load_into_mem")
        self.json_format = config.get("data", "json_format")

        if self.load_mem:
            self.data = []
            for filename in self.file_list:
                if self.json_format == "single":
                    self.data = self.data + json.load(open(filename, "r", encoding=encoding))
                else:
                    f = open(filename, "r", encoding=encoding)
                    for line in f:
                        tmp = self.check(line)
                        if tmp is None:
                            continue
                        self.data.append(tmp)
        else:
            print('gg le ')
            gg


    def check(self, data):
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
    

    def get_file_id(self, item):
        l = 0
        r = len(self.prefix_file_cnt)
        while l + 1 != r:
            m = (l + r) // 2
            if self.prefix_file_cnt[m] > item:
                l = m
            else:
                r = m

        return l


    def __getitem__(self, item):
        if self.load_mem:
            return self.data[item]
        else:
            which = self.get_file_id(item)
            if which == 0:
                idx = item
            else:
                idx = item - self.prefix_file_cnt[which - 1]

            if self.json_format == "single":
                if self.temp_data["file_id"] != which:
                    self.temp_data = {
                        "data": json.load(open(self.file_list[which], "r", encoding=self.encoding)),
                        "file_id": 0
                    }

                return self.temp_data["data"][idx]

            else:
                if self.temp_file_list[which]["cnt"] > idx:
                    self.temp_file_list[which] = {
                        "file": open(self.file_list[which], "r", encoding=self.encoding),
                        "cnt": 0
                    }

                delta = idx - self.temp_file_list[which]["cnt"]
                self.temp_file_list[which]["file"].readlines(delta)

                data = json.loads(self.temp_file_list[which]["file"].readline())
                self.temp_file_list[which]["cnt"] = idx + 1

                return data

    def __len__(self):
        if self.load_mem:
            return len(self.data)
        else:
            return self.total
