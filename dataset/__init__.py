from .nlp.JsonFromFiles import JsonFromFilesDataset
from .others.FilenameOnly import FilenameOnlyDataset
from .cv.ImageFolder import ImageFolderDataset

dataset_list = {
    "ImageFolder": ImageFolderDataset,
    "JsonFromFiles": JsonFromFilesDataset,
    "FilenameOnly": FilenameOnlyDataset
}
