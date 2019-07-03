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

    batch_size = config.getint("train", "batch_size")

    logger.info("Training start....")

    # print('** start training here! **')
    # print('----------------|----------TRAIN-----------|----------VALID-----------|----------------|')
    # print('  lr    epoch   |   loss           evalu   |   loss           evalu   |      time      | Forward num')
    # print('----------------|--------------------------|--------------------------|----------------|')
    # start = timer()

    acc_result = None
    total_loss = 0
    cnt = 0

    with tqdm(dataset, desc="Valid Iteration") as T:
        for step, data in enumerate(T):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            results = model(data, config, gpu_list, acc_result)

            loss, acc_result = results["loss"], results["acc_result"]
            total_loss += loss
            cnt += 1

        output_info = output_function(acc_result, config)
        T.set_postfix(loss=float(total_loss) / (step + 1), output=output_info)
        # tqdm.write("Epoch %d\tIter %d\t\tLoss %.3f\t\t%s" % (epoch, cnt, float(total_loss) / cnt, output_info))

    model.train()
