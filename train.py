from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import torch
import torch.utils.data
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from lib.utils.opts import opts
from lib.utils.logger import Logger
from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.coco_bhdata import COCO
from lib.Trainer.ctdet import CtdetTrainer


def main(opt):
    torch.manual_seed(opt.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0

    if len(opt.gpus) > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    os.environ["NCCL_P2P_DISABLE"] = "1"

    val_intervals = opt.val_intervals

    DataVal = COCO(opt, "val")
    DataTrain = COCO(opt, "train")

    if opt.test_real_data:
        DataTrain = ConcatDataset([DataTrain, DataVal])
        DataVal = COCO(opt, "test")
        print("Using real data for training")
    else:
        print("Using synthetic data for training")

    train_sampler = (
        DistributedSampler(DataTrain)
        if len(opt.gpus) > 1
        else torch.utils.data.RandomSampler(DataTrain)
    )
    val_sampler = (
        DistributedSampler(DataVal, shuffle=False)
        if len(opt.gpus) > 1
        else torch.utils.data.RandomSampler(DataVal)
    )

    val_loader = torch.utils.data.DataLoader(
        DataVal,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,
    )

    train_loader = torch.utils.data.DataLoader(
        DataTrain,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    print("Creating model...")
    head = {"hm": DataTrain.num_classes, "wh": 2, "reg": 2}
    model = get_det_net(head, opt.model_name)

    print(opt.model_name)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.save_results_dir, exist_ok=True)

    if opt.load_model != "":
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
        )

    trainer = CtdetTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    if local_rank == 0:
        logger = Logger(opt)
        print("Starting training...")

    best = -1

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        if len(opt.gpus) > 1:
            train_sampler.set_epoch(epoch)

        log_dict_train, _ = trainer.train(epoch, train_loader)

        if local_rank == 0:
            logger.write(f"epoch: {epoch} |")

            save_model(
                os.path.join(opt.save_dir, "model_last.pth"),
                epoch,
                model,
                optimizer,
            )

            for k, v in log_dict_train.items():
                logger.write(f"{k} {v:8f} | ")

            if val_intervals > 0 and epoch % val_intervals == 0:
                save_model(
                    os.path.join(opt.save_dir, f"model_{epoch}.pth"),
                    epoch,
                    model,
                    optimizer,
                )

        with torch.no_grad():
            log_dict_val, preds, stats = trainer.multi_gpu_val(
                epoch, val_loader, DataVal
            )

        if local_rank == 0:
            for k, v in log_dict_val.items():
                logger.write(f"{k} {v:8f} | ")

            logger.write("eval results: ")
            for k in stats.tolist():
                logger.write(f"{k:8f} | ")

            if log_dict_val["ap50"] > best:
                best = log_dict_val["ap50"]
                save_model(
                    os.path.join(opt.save_dir, "model_best.pth"),
                    epoch,
                    model,
                )
            else:
                save_model(
                    os.path.join(opt.save_dir, "model_last.pth"),
                    epoch,
                    model,
                    optimizer,
                )

            logger.write("\n")

            if epoch in opt.lr_step:
                save_model(
                    os.path.join(opt.save_dir, f"model_{epoch}.pth"),
                    epoch,
                    model,
                    optimizer,
                )

                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print("Drop LR to", lr)

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

    logger.close()


if __name__ == "__main__":
    opt = opts().parse()
    main(opt)