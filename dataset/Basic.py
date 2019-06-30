class Dataset(DATA.Dataset):
    # 数据集，从文件夹读入.mat文件，存储是[{}]的形式，由formatter完成数据的修饰
    def __init__(self, config, data_source):
        self.data = []
        self.data_source = data_source
        self.key_list = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVR", "aVL", "aVF"]
        self.load_mem = config.getboolean("reader", "load_mem")
        if self.load_mem:
            file_list = os.listdir(data_source)
            for file in file_list:
                raw_data = json.load(open(os.path.join(data_source, file), "r"))
                if "label" in raw_data.keys():
                    self.data.append(
                        {"data": [], "label": raw_data["label"], "gender": raw_data["sex"], "age": raw_data["age"],
                         "file": file[:-4], "feature": raw_data["feature"]})
                else:
                    self.data.append(
                        {"data": [], "label": [0], "gender": raw_data["sex"], "age": raw_data["age"],
                         "file": file[:-4], "feature": raw_data["feature"]})
                length_record = []
                for key in self.key_list:
                    if len(raw_data[key]) == 1:
                        length_record.append(len(raw_data[key][0]))
                        self.data[-1]["data"].append(raw_data[key][0])
                    else:
                        gg
                length_record = np.array(length_record)
                for item in length_record:
                    if item != np.mean(length_record):
                        print(file)
                        print(length_record)
                        self.data = self.data[:-1]
                        break
        else:
            self.file_list = os.listdir(data_source)
            self.data = []
            for a in range(0, len(self.file_list)):
                self.data.append(None)

    def __getitem__(self, item):
        if self.load_mem:
            return self.data[item]
        else:
            if self.data[item] is None:
                raw_data = json.load(open(os.path.join(self.data_source, self.file_list[item]), "r"))
                if "label" in raw_data.keys():
                    data = {"data": [], "label": raw_data["label"], "gender": raw_data["sex"], "age": raw_data["age"],
                            "file": self.file_list[item][:-4], "feature": raw_data["feature"]}
                else:
                    data = {"data": [], "label": [0], "gender": raw_data["sex"], "age": raw_data["age"],
                            "file": self.file_list[:-4], "feature": raw_data["feature"]}
                length_record = []
                for key in self.key_list:
                    if len(raw_data[key]) == 1:
                        length_record.append(len(raw_data[key][0]))
                        data["data"].append(raw_data[key][0])
                self.data[item] = data
            return self.data[item]

    def __len__(self):
        return len(self.data)
