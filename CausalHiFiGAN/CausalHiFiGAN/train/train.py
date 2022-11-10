import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from ..models.generator import Generator
from ..models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from ..tools.melspectrogram import MelSpectrogramExtractorForLoss
from .dataset import MelDataset
from .losses import feature_loss, generator_loss, discriminator_loss
from .utils import scan_checkpoint, load_checkpoint, save_checkpoint
from ..tools.file_io import load_json, load_list
from ..tools.plot import MelSpectrogramPlotter

torch.backends.cudnn.benchmark = True


def train(path_dir_list=Path("./data/list"),
          path_dir_checkpoint=Path("./checkpoint"),
          path_dir_log=Path("./log"),
          path_config="./config_v1.json",
          path_dir_basemodel=None):
    print("--- train ---")

    # prepare directory

    path_dir_checkpoint.mkdir(exist_ok=1)
    path_dir_log.mkdir(exist_ok=1)

    # prepare device

    h = load_json(path_config)

    if mp.get_start_method() == 'fork' and h.num_workers != 0:
        mp.set_start_method('spawn', force=True)

    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    device = torch.device(h.device)

    # prepare model

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    print(generator)

    cp_g = scan_checkpoint(path_dir_checkpoint, 'g_')
    cp_do = scan_checkpoint(path_dir_checkpoint, 'do_')
    if cp_g is not None and cp_do is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        step = state_dict_do['step']
        epoch = state_dict_do['epoch']
        del state_dict_g
        del state_dict_do
    elif path_dir_basemodel is not None and path_dir_basemodel.is_dir():
        cp_g = scan_checkpoint(path_dir_basemodel, 'g_')
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        step = 0
        epoch = 0
        del state_dict_g
    else:
        step = 0
        epoch = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=step - 1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=step - 1)

    # prepare dataset

    dataset_train = MelDataset(path_dir_list / "train.txt",
                               h,
                               h.segment_size,
                               generator.tail,
                               randomize=True)
    dataset_valid = MelDataset(path_dir_list / "valid.txt",
                               h,
                               h.segment_size_valid,
                               generator.tail,
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

    # prepare function

    f_melspe = MelSpectrogramExtractorForLoss(n_fft=h.n_fft,
                                              win_size=h.win_size,
                                              hop_size=h.hop_size,
                                              sampling_rate=h.sampling_rate,
                                              num_mels=h.num_mels,
                                              fmin=h.fmin,
                                              fmax=h.fmax,
                                              device=device)

    # prepare tensorboard

    sw = SummaryWriter(path_dir_log)

    list_path_sample = load_list(path_dir_list / "valid_sample.txt")
    list_index_sample = [dataset_valid.list_path_wav.index(path_sample) for path_sample in list_path_sample]
    list_index_sample = [(i // h.batch_size_valid, i % h.batch_size_valid) for i in list_index_sample]
    count_sample = 0
    len_list_sample = len(list_index_sample)

    plot_melspe = MelSpectrogramPlotter(h)

    # training

    generator.train()
    mpd.train()
    msd.train()

    while step < h.training_iterations:
        epoch += 1

        for batch in dataloader_train:
            step += 1
            start_time = time.time()

            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                y_mel = f_melspe(y)
            y = y.unsqueeze(1)

            y_hat = generator(x)
            y_hat_mel = f_melspe(y_hat.squeeze(1))

            optim_d.zero_grad(set_to_none=True)

            # MPD
            r_mpd_real, r_mpd_fake, _, _ = mpd(y, y_hat.detach())
            loss_mpd_real, loss_mpd_fake = discriminator_loss(r_mpd_real, r_mpd_fake)

            # MSD
            r_msd_real, r_msd_fake, _, _ = msd(y, y_hat.detach())
            loss_msd_real, loss_msd_fake = discriminator_loss(r_msd_real, r_msd_fake)

            loss_d = loss_msd_real + loss_msd_fake + loss_mpd_real + loss_mpd_fake
            loss_d.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad(set_to_none=True)

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_hat_mel)

            _, r_mpd_fake, fmap_mpd_real, fmap_mpd_fake = mpd(y, y_hat)
            _, r_msd_fake, fmap_msd_real, fmap_msd_fake = msd(y, y_hat)
            loss_fm_mpd = feature_loss(fmap_mpd_real, fmap_mpd_fake)
            loss_fm_msd = feature_loss(fmap_msd_real, fmap_msd_fake)
            loss_g_mpd = generator_loss(r_mpd_fake)
            loss_g_msd = generator_loss(r_msd_fake)
            loss_g = loss_g_msd + loss_g_mpd + (loss_fm_msd + loss_fm_mpd) * 2 + loss_mel * 45

            loss_g.backward()
            optim_g.step()

            # checkpointing
            if step % h.checkpoint_interval == 0 and step != 0:
                checkpoint_path = "{}/g_{:08d}".format(path_dir_checkpoint, step)
                save_checkpoint(checkpoint_path,
                                {'generator': generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(path_dir_checkpoint, step)
                save_checkpoint(checkpoint_path,
                                {'mpd': mpd.state_dict(),
                                 'msd': msd.state_dict(),
                                 'optim_g': optim_g.state_dict(),
                                 'optim_d': optim_d.state_dict(),
                                 'epoch': epoch,
                                 'step': step})

            # Tensorboard summary logging
            if step % h.summary_interval == 0:
                print('Step : {:d}, Mel-Spec. loss : {:4.3f}'.
                      format(step, loss_mel))

                losses = {"train/loss_d": loss_d.item(),
                          "train/loss_msd_real": loss_msd_real.item(),
                          "train/loss_msd_fake": loss_msd_fake.item(),
                          "train/loss_mpd_real": loss_mpd_real.item(),
                          "train/loss_mpd_fake": loss_mpd_fake.item(),
                          "train/loss_g": loss_g.item(),
                          "train/loss_g_msd": loss_g_msd.item(),
                          "train/loss_g_mpd": loss_g_mpd.item(),
                          "train/loss_g_feats_match_msd": loss_fm_msd.item(),
                          "train/loss_g_feats_match_mpd": loss_fm_mpd.item(),
                          "train/loss_g_melspec": loss_mel.item()}
                for lab, loss in losses.items():
                    sw.add_scalar(lab, loss, step)

                sw.add_scalar("train/lr", scheduler_g.get_last_lr()[0], step)

            # validation
            if step % h.validation_interval == 0:
                generator.eval()
                torch.cuda.empty_cache()
                loss_mel = 0
                with torch.no_grad():
                    i = 0
                    for j, batch in enumerate(dataloader_valid):
                        x, y = batch
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        with torch.no_grad():
                            y_mel = f_melspe(y)

                        y_hat = generator(x)
                        y_hat_mel = f_melspe(y_hat.squeeze(1))
                        loss_mel += F.l1_loss(y_mel, y_hat_mel).item()

                        if i < len_list_sample:
                            if j == list_index_sample[i][0]:
                                name_sample = f"{list_path_sample[i].parent.stem}_{list_path_sample[i].stem}"
                                if count_sample < len_list_sample:
                                    y = y.unsqueeze(1)
                                    sw.add_audio('target/wav_{}'.format(name_sample),
                                                 y[list_index_sample[i][1]].cpu(), 0, h.sampling_rate)
                                    sw.add_figure('target/spe_{}'.format(name_sample),
                                                  plot_melspe(y_mel[list_index_sample[i][1]].cpu().numpy()), 0)
                                    count_sample += 1

                                sw.add_audio('generated/wav_{}'.format(name_sample),
                                             y_hat[list_index_sample[i][1]].cpu(), step, h.sampling_rate)
                                sw.add_figure('generated/spe_{}'.format(name_sample),
                                              plot_melspe(y_hat_mel[list_index_sample[i][1]].cpu().numpy()), step)
                                i += 1

                    val_err = loss_mel / len(dataloader_valid)
                    sw.add_scalar("valid/loss_g_melspec", val_err, step)
                generator.train()

            scheduler_g.step()
            scheduler_d.step()

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start_time)))
