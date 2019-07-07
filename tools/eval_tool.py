import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def valid(model, dataset, epoch, writer, config, gpu_list, output_function):
    model.eval()

    ncols = None
    try:
        ncol = config.getint("output", "tqdm_ncols")
    except Exception as e:
        ncol = None

    # logger.info("Training start....")

    # print('** start training here! **')
    # print('----------------|----------TRAIN-----------|----------VALID-----------|----------------|')
    # print('  lr    epoch   |   loss           evalu   |   loss           evalu   |      time      | Forward num')
    # print('----------------|--------------------------|--------------------------|----------------|')
    # start = timer()

    acc_result = None
    total_loss = 0
    cnt = 0

    with tqdm(dataset, desc="Valid Epoch %d" % epoch, ncols=ncol) as T:
        for step, data in enumerate(T):
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

            if step == len(dataset) - 1:
                output_info = output_function(acc_result, config)
                T.set_postfix(loss=float(total_loss) / cnt, output=output_info)

        # tqdm.write("Epoch %d\tIter %d\t\tLoss %.3f\t\t%s" % (epoch, cnt, float(total_loss) / cnt, output_info))

        print("")
        print("")

        writer.add_scalar(config.get("output", "model_name") + "_eval_epoch", float(total_loss) / (step + 1),
                          epoch)

    model.train()
