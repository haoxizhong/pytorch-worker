import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from tools.eval_tool import valid

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = config.get("output", "model_path")

    trained_epoch = parameters["trained_epoch"]
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    output_function = parameters["output_function"]

    # os.makedirs(os.path.join(config.get("output", "tensorboard_path")), exist_ok=True)

    # if trained_epoch == 0:
    #    shutil.rmtree(
    #        os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    writer = SummaryWriter(
        os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
        config.get("output", "model_name"))

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    # print('** start training here! **')
    # print('----------------|----------TRAIN-----------|----------VALID-----------|----------------|')
    # print('  lr    epoch   |   loss           evalu   |   loss           evalu   |      time      | Forward num')
    # print('----------------|--------------------------|--------------------------|----------------|')
    # start = timer()

    for epoch_num in enumerate(tqdm(range(trained_epoch, epoch), desc="Epoch")):
        current_epoch = epoch_num[0]

        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        # for step,data in enumerate(dataset):
        with tqdm(dataset, desc="Train Iteration") as T:
            output_info = ""
            for step, data in enumerate(T):

                if step < 1245:
                    continue
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        if len(gpu_list) > 0:
                            data[key] = Variable(data[key].cuda())
                        else:
                            data[key] = Variable(data[key])

                optimizer.zero_grad()

                results = model(data, config, gpu_list, acc_result)

                loss, acc_result = results["loss"], results["acc_result"]
                total_loss += loss

                loss.backward()
                optimizer.step()

                if step % output_time == 0:
                    output_info = output_function(acc_result, config)

                    # tqdm.write("Epoch %d\tIter %d\t\tLoss %.3f\t\t%s" % (
                    #    current_epoch, step, float(total_loss) / (step + 1), output_info))

                T.set_postfix(loss=float(total_loss) / (step + 1), output=output_info)

        checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function)