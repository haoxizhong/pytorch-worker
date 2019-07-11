from .nlp.JsonFromFiles import JsonFromFilesDataset
from .others.FilenameOnly import FilenameOnlyDataset
#from .cv.ImageFromJson import ImageFromJsonDataset
from .demo.JsonFromMem import JsonFromMemDataset


dataset_list = {
#    "ImageFromJson": ImageFromJsonDataset,
    "JsonFromFiles": JsonFromFilesDataset,
    "FilenameOnly": FilenameOnlyDataset,
    "JsonFromMem": JsonFromMemDataset
}
