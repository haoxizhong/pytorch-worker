from torch.utils.data import Dataset
import cv2

from tools.dataset_tools import dfs_search


class ImageFolderDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive")

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(name, recursive)
        self.file_list.sort()

        self.load_mem = config.getboolean("data", "load_into_mem")
        if self.load_mem:
            self.data = []
            for filename in self.file_list:
                self.data.append(cv2.imread(filename))

    def __getitem__(self, item):
        if self.load_mem:
            return self.data[item]
        else:
            filename = self.file_list[item]
            return cv2.imread(filename)

    def __len__(self):
        return len(self.file_list)
