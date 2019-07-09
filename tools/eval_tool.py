import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def valid(model, dataset, epoch, writer, config, gpu_list, output_function):
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "valid")

        loss, acc_result = results["loss"], results["acc_result"]
        total_loss += loss
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            print("%d\t%s\t%d/%d%s\t%s/%s\t%s" % (
                epoch, "valid", step + 1, total_len, more, gen_time_str(delta_t),
                gen_time_str(delta_t * (total_len - step - 1) / (step + 1)), output_info), end='\r')

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = output_function(acc_result, config)
    print("%d\t%s\t%d/%d%s\t%s/%s\t%s" % (
        epoch, "valid", step + 1, total_len, more, gen_time_str(delta_t),
        gen_time_str(delta_t * (total_len - step - 1) / (step + 1)), output_info))

    writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                      epoch)

    model.train()
