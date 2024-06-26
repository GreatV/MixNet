import paddle
import os
import gc
import time
import numpy as np
from dataset import (
    SynthText,
    TotalText,
    Ctw1500Text,
    TD500HUSTText,
    ArtTextJson,
    MLTTextJson,
    TotalText_mid,
    Ctw1500Text_mid,
    TD500HUSTText_mid,
    ALLTextJson,
    ArtTextJson_mid,
)
from network.loss import TextLoss, knowledge_loss
from network.textnet import TextNet
from util.augmentation import Augmentation
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from cfglib.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary
from util.shedule import FixLR

lr = None
train_step = 0


def save_model(model, epoch, lr, optimzer):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
    save_path = os.path.join(
        save_dir, "MixNet_{}_{}.pth".format(model.backbone_name, epoch)
    )
    print("Saving to {}.".format(save_path))
    state_dict = {
        "lr": lr,
        "epoch": epoch,
        "model": model.state_dict() if not cfg.mgpu else model.module.state_dict(),
    }
    paddle.save(obj=state_dict, path=save_path)


def load_model(model, model_path):
    print("Loading from {}".format(model_path))
    state_dict = paddle.load(path=model_path)
    try:
        model.set_state_dict(state_dict=state_dict["model"])
    except RuntimeError as e:
        print("Missing key in state_dict, try to load with strict = False")
        model.set_state_dict(state_dict=state_dict["model"], use_structured_name=False)
        print(e)


def _parse_data(inputs):
    input_dict = {}
    inputs = list(map(lambda x: to_device(x), inputs))
    input_dict["img"] = inputs[0]
    input_dict["train_mask"] = inputs[1]
    input_dict["tr_mask"] = inputs[2]
    input_dict["distance_field"] = inputs[3]
    input_dict["direction_field"] = inputs[4]
    input_dict["weight_matrix"] = inputs[5]
    input_dict["gt_points"] = inputs[6]
    input_dict["proposal_points"] = inputs[7]
    input_dict["ignore_tags"] = inputs[8]
    if cfg.embed:
        input_dict["edge_field"] = inputs[9]
    if cfg.mid:
        input_dict["gt_mid_points"] = inputs[9]
        input_dict["edge_field"] = inputs[10]
    return input_dict


def train(model, train_loader, criterion, scheduler, optimizer, epoch):
    global train_step
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    print("Epoch: {} : LR = {}".format(epoch, scheduler.get_lr()))
    for i, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_step += 1
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        loss_dict = criterion(input_dict, output_dict, eps=epoch + 1)
        loss = loss_dict["total_loss"]
        optimizer.clear_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            paddle.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=cfg.grad_clip
            )
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0) and epoch % 8 == 0:
            visualize_network_output(output_dict, input_dict, mode="train")
        if i % cfg.display_freq == 0:
            gc.collect()
            print_inform = "({:d} / {:d}) ".format(i, len(train_loader))
            for k, v in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            print(print_inform)
    if (
        cfg.exp_name == "Synthtext"
        or cfg.exp_name == "ALL"
        or cfg.exp_name == "preSynthMLT"
        or cfg.exp_name == "preALL"
    ):
        print("save checkpoint for pretrain weight. ")
        if epoch % cfg.save_freq == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
    elif (
        cfg.exp_name == "MLT2019" or cfg.exp_name == "ArT" or cfg.exp_name == "MLT2017"
    ):
        if epoch < 10 and cfg.max_epoch >= 200:
            if epoch % (2 * cfg.save_freq) == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
        elif epoch % cfg.save_freq == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
    elif epoch % cfg.save_freq == 0 and epoch > 150:
        save_model(model, epoch, scheduler.get_lr(), optimizer)
    print("Training Loss: {}".format(losses.avg))


def knowledgetrain(
    model,
    knowledge,
    train_loader,
    criterion,
    know_criterion,
    scheduler,
    optimizer,
    epoch,
):
    global train_step
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    print("Epoch: {} : LR = {}".format(epoch, scheduler.get_lr()))
    for i, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_step += 1
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        output_know = knowledge(input_dict, knowledge=True)
        loss_dict = criterion(input_dict, output_dict, eps=epoch + 1)
        loss = loss_dict["total_loss"]
        know_loss = know_criterion(
            output_dict["image_feature"], output_know["image_feature"]
        )
        loss = loss + know_loss
        loss_dict["know_loss"] = know_loss
        optimizer.clear_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            paddle.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=cfg.grad_clip
            )
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0) and epoch % 8 == 0:
            visualize_network_output(output_dict, input_dict, mode="train")
        if i % cfg.display_freq == 0:
            gc.collect()
            print_inform = "({:d} / {:d}) ".format(i, len(train_loader))
            for k, v in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            print(print_inform)
    if (
        cfg.exp_name == "Synthtext"
        or cfg.exp_name == "ALL"
        or cfg.exp_name == "preSynthMLT"
        or cfg.exp_name == "preALL"
    ):
        print("save checkpoint for pretrain weight. ")
        if epoch % cfg.save_freq == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
    elif (
        cfg.exp_name == "MLT2019" or cfg.exp_name == "ArT" or cfg.exp_name == "MLT2017"
    ):
        if epoch < 10 and cfg.max_epoch >= 200:
            if epoch % (2 * cfg.save_freq) == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
        elif epoch % cfg.save_freq == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
    elif epoch % cfg.save_freq == 0 and epoch > 150:
        save_model(model, epoch, scheduler.get_lr(), optimizer)
    print("Training Loss: {}".format(losses.avg))


def main():
    global lr
    if cfg.exp_name == "Totaltext":
        trainset = TotalText(
            data_root="data/total-text-mat",
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "Totaltext_mid":
        trainset = TotalText_mid(
            data_root="data/total-text-mat",
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "Synthtext":
        trainset = SynthText(
            data_root="../FAST/data/SynthText",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "Ctw1500":
        trainset = Ctw1500Text(
            data_root="data/ctw1500",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "Ctw1500_mid":
        trainset = Ctw1500Text_mid(
            data_root="data/ctw1500",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "TD500HUST":
        trainset = TD500HUSTText(
            data_root="data/",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "TD500HUST_mid":
        trainset = TD500HUSTText_mid(
            data_root="data/",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "ArT":
        trainset = ArtTextJson(
            data_root="data/ArT",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "ArT_mid":
        trainset = ArtTextJson_mid(
            data_root="data/ArT",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "preSynthMLT":
        trainset = MLTTextJson(
            is_training=True,
            load_memory=False,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "preALL":
        trainset = ALLTextJson(
            is_training=True,
            load_memory=False,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        valset = None
    elif cfg.exp_name == "ALL":
        trainset_SynthMLT = MLTTextJson(
            is_training=True,
            load_memory=False,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        trainset_SynthText = SynthText(
            data_root="../FAST/data/SynthText",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        trainset_totaltext = TotalText(
            data_root="data/total-text-mat",
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        trainset_TD500 = TD500HUSTText(
            data_root="data/",
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds),
        )
        trainset = paddle.io.ConcatDataset(
            datasets=[
                trainset_SynthText,
                trainset_SynthMLT,
                trainset_totaltext,
                trainset_TD500,
            ]
        )
        valset = None
    else:
        print("dataset name is not correct")
    train_loader = paddle.io.DataLoader(
        dataset=trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    if (
        cfg.exp_name == "Synthtext"
        or cfg.exp_name == "ALL"
        or cfg.exp_name == "preSynthMLT"
    ):
        print("save checkpoint for pretrain weight. ")
    model = TextNet(backbone=cfg.net, is_training=True)
    model = model.to(cfg.device)
    if cfg.know:
        know_model = TextNet(backbone=cfg.knownet, is_training=False)
        load_model(know_model, cfg.know_resume)
        know_model.eval()
        know_model.stop_gradient = not False
    if cfg.exp_name == "TD500HUST" or cfg.exp_name == "Ctw1500":
        criterion = TextLoss_ctw()
    else:
        criterion = TextLoss()
    if cfg.mgpu:
        model = paddle.DataParallel(layers=model)
    if cfg.cuda:
        pass
    if cfg.resume:
        load_model(model, cfg.resume)
    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == "Synthtext":
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=lr, weight_decay=0.0
        )
    else:
        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr, momentum=moment, parameters=model.parameters()
        )
    if cfg.exp_name == "Synthtext":
        scheduler = FixLR(optimizer)
    else:
        tmp_lr = paddle.optimizer.lr.StepDecay(
            step_size=50, gamma=0.9, learning_rate=optimizer.get_lr()
        )
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
    print("Start training MixNet.")
    for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
        scheduler.step()
        if cfg.know:
            know_criterion = knowledge_loss(T=5)
            knowledgetrain(
                model,
                know_model,
                train_loader,
                criterion,
                know_criterion,
                scheduler,
                optimizer,
                epoch,
            )
        else:
            train(model, train_loader, criterion, scheduler, optimizer, epoch)
    print("End.")
    if paddle.device.cuda.device_count() >= 1:
        paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2022)
    paddle.seed(seed=2022)
    option = BaseOptions()
    args = option.initialize()
    update_config(cfg, args)
    print_config(cfg)
    main()
