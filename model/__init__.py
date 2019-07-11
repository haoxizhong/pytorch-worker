from .nlp.BasicBert import BasicBert
from .demo.MultiTaskBert import MultiTaskBert


model_list = {
    "BasicBert": BasicBert,
    "MultiTaskBert": MultiTaskBert
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
