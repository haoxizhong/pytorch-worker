from .nlp.JsonFromFiles import JsonFromFilesDataset
from .others.FilenameOnly import FilenameOnlyDataset
from .cv.ImageFromJson import ImageFromJsonDataset

dataset_list = {
    "ImageFromJson": ImageFromJsonDataset,
    "JsonFromFiles": JsonFromFilesDataset,
    "FilenameOnly": FilenameOnlyDataset
}
