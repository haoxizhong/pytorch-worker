import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)


name2key = {
    'macro_precision': 'mac_p',
    'micro_precision': 'mic_p',
    'macro_recall': 'mac_r',
    'micro_recall': 'mic_r',
    'macro_f1': 'mac_f1',
    'micro_f1': 'mic_f1'
}
def multi_task_function(data, config, *args, **params):
    which = config.get('output', 'output_value').replace(' ', '').split(',')
    result = {}
    global name2key

    for key in data:
        result[key] = {}
        temp = gen_micro_macro_result(data[key])
        for name in which:
            result[key][name2key[name]] = temp[name]

    return json.dumps(result, sort_keys = True)



