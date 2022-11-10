import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter

from .dataset import SpeF0Dataset
from .schedule import CosExp
from ..models.jdcnet import JDCNet
from ..tools.plot import F0Plotter

from CausalHiFiGAN.tools.file_io import load_json, load_list

torch.backends.cudnn.benchmark = True


def train(path_dir_list=Path("./data/list"),
          path_dir_param=Path("./data/param"),
          path_dir_checkpoint=Path("./checkpoint"),
          path_dir_log=Path("./log"),
          path_config="./config.json",
          path_config_vocoder="../CausalHiFiGAN/config_v1.json"):
    print("--- train ---")

    # prepare directory

    path_dir_checkpoint.mkdir(exist_ok=1)
    path_dir_log.mkdir(exist_ok=1)

    # load config

    h = load_json(path_config)
    hv = load_json(path_config_vocoder)

    # prepare device

    if multiprocessing.get_start_method() == 'fork' and h.num_workers != 0:
        multiprocessing.set_start_method('spawn', force=True)

    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    device = torch.device(h.device)

    # prepare model

    recognizer = JDCNet()
    recognizer = recognizer.to(device)

    path_cp = path_dir_checkpoint / "recognizer_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        recognizer.load_state_dict(cp)
        del cp
        print(f"loaded {path_cp}")

    # prepare dataset

    dataset_train = SpeF0Dataset(path_dir_list / "train.txt",
                                 path_dir_param / "stats.json",
                                 h, hv,
                                 h.segment_size,
                                 randomize=True)
    dataset_valid = SpeF0Dataset(path_dir_list / "valid.txt",
                                 path_dir_param / "stats.json",
                                 h, hv,
                                 h.segment_size_valid,
                                 randomize=False)

    dataloader_train = DataLoader(dataset_train,
                                  h.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=h.num_workers,
                                  pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid,
                                  h.batch_size_valid,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=h.num_workers,
                                  pin_memory=True)

    # prepare optimizer

    optimizer = torch.optim.RAdam(recognizer.parameters(),
                                  h.lr,
                                  h.betas,
                                  h.eps,
                                  h.weight_decay)

    cosexp = CosExp(h.epoch_warmup * len(dataloader_train),
                    h.epoch_switch * len(dataloader_train),
                    h.weight_lr_initial,
                    h.weight_lr_final)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosexp)

    path_cp = path_dir_checkpoint / "state_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        recognizer.load_state_dict(cp["recognizer"])
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        epoch = cp["epoch"]
        step = cp["step"]
        loss_best = cp["loss_best"]
        del cp
        print(f"loaded {path_cp}")
    else:
        epoch = 0
        step = 0
        loss_best = 1.0

    # prepare tensorboard

    sw = SummaryWriter(path_dir_log)

    list_path_sample = load_list(path_dir_list / "valid_sample.txt")
    list_index_sample = [dataset_valid.list_path_wav.index(path_sample) for path_sample in list_path_sample]
    list_index_sample = [(i // h.batch_size_valid, i % h.batch_size_valid) for i in list_index_sample]
    len_list_sample = len(list_index_sample)

    plot_f0 = F0Plotter(h, hv)

    # start training

    while(epoch < h.epochs):
        epoch += 1
        print(f"--- epoch {epoch} train ---")

        # train

        for batch in tqdm(dataloader_train):
            step += 1

            # forward

            spe, f0, vuv = [item.to(device) for item in batch]

            f0_h, vuv_h = recognizer(spe)
            loss_f0 = F.smooth_l1_loss(f0_h * vuv, f0)
            loss_vuv = F.binary_cross_entropy(vuv_h, vuv)
            loss = loss_f0 + h.weight_loss_vuv * loss_vuv

            # backward

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # write log

            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            scheduler.step()

            sw.add_scalar("train/loss_f0", loss_f0, step)
            sw.add_scalar("train/loss_vuv", loss_vuv, step)

        # valid

        if epoch % h.valid_interval == 0:
            print(f"--- epoch {epoch} valid ---")

            recognizer.eval()
            with torch.no_grad():
                loss_f0 = 0.0
                loss_vuv = 0.0
                i = 0
                for j, batch in tqdm(enumerate(dataloader_valid)):

                    # forward

                    spe, f0, vuv = [item.to(device) for item in batch]

                    f0_h, vuv_h = recognizer(spe)
                    loss_f0 += F.smooth_l1_loss(f0_h * vuv, f0)
                    loss_vuv += F.binary_cross_entropy(vuv_h, vuv)

                    # visualize samples

                    if i < len_list_sample:
                        if j == list_index_sample[i][0]:
                            name_sample = f"{list_path_sample[i].parent.stem}_{list_path_sample[i].stem}"

                            f0_np = f0[list_index_sample[i][1]].cpu().numpy()
                            vuv_np = vuv[list_index_sample[i][1]].cpu().numpy()
                            f0_h_np = f0_h[list_index_sample[i][1]].cpu().numpy()
                            vuv_h_np = vuv_h[list_index_sample[i][1]].cpu().numpy()
                            f0_np = np.where(vuv_np == 0.0, np.nan, f0_np)
                            f0_h_np = np.where(vuv_h_np < 0.5, np.nan, f0_h_np)
                            sw.add_figure('predicted/f0_{}'.format(name_sample),
                                          plot_f0(f0_np, f0_h_np), step)
                            i += 1

                loss_f0 /= len(dataloader_valid)
                loss_vuv /= len(dataloader_valid)

                # write log
                sw.add_scalar("valid/loss_f0", loss_f0, step)
                sw.add_scalar("valid/loss_vuv", loss_vuv, step)

                # save state

                torch.save(
                    {"recognizer": recognizer.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "step": step,
                     "loss_best": loss_best},
                    path_dir_checkpoint / "state_latest.cp")

                print("saved state_latest.cp")

                if loss_f0 <= loss_best:
                    loss_best = loss_f0
                    torch.save(
                        recognizer.state_dict(),
                        path_dir_checkpoint / "recognizer_best.cp")
                    print("saved recognizer_best.cp")

            recognizer.train()
