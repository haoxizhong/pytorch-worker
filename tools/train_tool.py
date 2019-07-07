import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from tools.eval_tool import valid
import shutil

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
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
    ncols = None
    try:
        ncol = config.getint("output", "tqdm_ncols")
    except Exception as e:
        ncol = None

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"]
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                exist_ok=True)

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                           config.get("output", "model_name"))

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    for epoch_num in enumerate(tqdm(range(trained_epoch, epoch), desc="Epoch", ncols=ncol)):
        current_epoch = epoch_num[1]

        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        # for step,data in enumerate(dataset):
        with tqdm(dataset, desc="Train Epoch %d" % current_epoch, ncols=ncol) as T:
            output_info = ""
            for step, data in enumerate(T):
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        if len(gpu_list) > 0:
                            data[key] = Variable(data[key].cuda())
                        else:
                            data[key] = Variable(data[key])

                optimizer.zero_grad()

                results = model(data, config, gpu_list, acc_result, "train")

                loss, acc_result = results["loss"], results["acc_result"]
                total_loss += loss

                loss.backward()
                optimizer.step()

                if step % output_time == 0:
                    output_info = output_function(acc_result, config)

                    # tqdm.write("Epoch %d\tIter %d\t\tLoss %.3f\t\t%s" % (
                    #    current_epoch, step, float(total_loss) / (step + 1), output_info))

                T.set_postfix(loss=float(total_loss) / (step + 1), output=output_info)
                global_step += 1
                writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss), global_step)

        print("")
        checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config,
                   global_step)
        writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1),
                          current_epoch)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function)
