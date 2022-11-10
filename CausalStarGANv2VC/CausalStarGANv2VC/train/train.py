import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter

from .dataset import SpeDataset, SpeValidDataset
from .standardizer import Standardizer
from ..tools.destandardizer import Destandardizer
from .losses import BCEWithLogitHingeLoss, f_r1reg, f_loss_f0
from ..models.generator import Generator, StyleEncoder
from ..models.discriminator import Discriminator, Classifier
from ..tools.speaker_dict import get_dict_speaker

from CausalHiFiGAN.tools.file_io import load_json, load_list
from CausalHiFiGAN.tools.plot import MelSpectrogramPlotter
from CNNConformer.tools.get_num_class import get_num_class
from CNNConformer.models.conformer import Conformer
from JDCNet.models.jdcnet import JDCNet
from StarGANv2VC.models.generator import Generator as Generator_noncausal

torch.backends.cudnn.benchmark = True


def train(path_dir_list=Path("./data/list"),
          path_dir_param=Path("./data/param"),
          path_dir_checkpoint=Path("./checkpoint"),
          path_dir_log=Path("./log"),
          path_config="./config.json",
          path_config_vocoder="../CausalHiFiGAN/config_v1.json",
          path_CNNConformer="../CNNConformer",
          path_JDCNet="../JDCNet",
          path_StarGANv2VC="../StarGANv2VC",
          epoch_StarGANv2VC=250):
    path_dir_param_cnnconformer = Path(f"{path_CNNConformer}/data/param")
    path_cp_cnnconformer = Path(f"{path_CNNConformer}/checkpoint/recognizer_best.cp")
    path_dir_param_jdcnet = Path(f"{path_JDCNet}/data/param")
    path_cp_jdcnet = Path(f"{path_JDCNet}/checkpoint/recognizer_best.cp")
    path_cp_generator_noncausal = Path(f"{path_StarGANv2VC}/checkpoint/generator_{str(epoch_StarGANv2VC).zfill(4)}.cp")
    path_cp_style_encoder_noncausal = Path(f"{path_StarGANv2VC}/checkpoint/style_encoder_{str(epoch_StarGANv2VC).zfill(4)}.cp")

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

    dict_speaker = get_dict_speaker(path_dir_list / "train")
    num_speaker = len(dict_speaker)

    generator = Generator(style_dim=h.style_dim).to(device)
    style_encoder = StyleEncoder(num_domains=num_speaker, style_dim=h.style_dim).to(device)
    discriminator = Discriminator(num_domains=num_speaker).to(device)
    classifier = Classifier(num_domains=num_speaker).to(device)

    # prepare optimizer

    optimizer_generator = torch.optim.RAdam(generator.parameters(),
                                            h.lr_generator,
                                            h.betas,
                                            h.eps,
                                            h.weight_decay)
    optimizer_style_encoder = torch.optim.RAdam(style_encoder.parameters(),
                                                h.lr_style_encoder,
                                                h.betas,
                                                h.eps,
                                                h.weight_decay)

    optimizer_discriminator = torch.optim.RAdam(discriminator.parameters(),
                                                h.lr_discriminator,
                                                h.betas,
                                                h.eps,
                                                h.weight_decay)

    optimizer_classifier = torch.optim.RAdam(classifier.parameters(),
                                             h.lr_classifier,
                                             h.betas,
                                             h.eps,
                                             h.weight_decay)

    path_cp = path_dir_checkpoint / "state_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        generator.load_state_dict(cp["generator"])
        style_encoder.load_state_dict(cp["style_encoder"])
        discriminator.load_state_dict(cp["discriminator"])
        classifier.load_state_dict(cp["classifier"])
        optimizer_generator.load_state_dict(cp["optimizer_generator"])
        optimizer_style_encoder.load_state_dict(cp["optimizer_style_encoder"])
        optimizer_discriminator.load_state_dict(cp["optimizer_discriminator"])
        optimizer_classifier.load_state_dict(cp["optimizer_classifier"])
        epoch = cp["epoch"]
        step = cp["step"]
        del cp
        print(f"loaded {path_cp}")
    else:
        epoch = 0
        step = 0

    # prepare dataset

    dataset_train = SpeDataset(path_dir_list / "train",
                               path_dir_param / "stats.json",
                               dict_speaker,
                               h, hv,
                               h.segment_size,
                               generator.tail * hv.hop_size,
                               randomize=True)
    dataset_valid = SpeValidDataset(path_dir_list / "valid",
                                    path_dir_list / "valid_target",
                                    path_dir_param / "stats.json",
                                    dict_speaker,
                                    h, hv,
                                    h.segment_size_valid,
                                    generator.tail * hv.hop_size,
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

    standardizer_recognizer_f0 = Standardizer(path_dir_param / "stats.json",
                                              path_dir_param_jdcnet / "stats.json")
    standardizer_recognizer_ppg = Standardizer(path_dir_param / "stats.json",
                                               path_dir_param_cnnconformer / "stats.json")
    destandardizer = Destandardizer(path_dir_param / "stats.json")

    # prepare model for loss

    recognizer_f0 = JDCNet().to(device)

    assert path_cp_jdcnet.exists()
    cp = torch.load(path_cp_jdcnet, map_location=lambda storage, loc: storage)
    recognizer_f0.load_state_dict(cp)
    del cp
    print(f"loaded {path_cp_jdcnet}")

    num_class = get_num_class(path_dir_param_cnnconformer / "phoneme.json")
    recognizer_ppg = Conformer(num_class).to(device)

    assert path_cp_cnnconformer.exists()
    cp = torch.load(path_cp_cnnconformer, map_location=lambda storage, loc: storage)
    recognizer_ppg.load_state_dict(cp)
    del cp
    print(f"loaded {path_cp_cnnconformer}")

    generator_noncausal = Generator_noncausal(style_dim=h.style_dim).to(device)
    assert path_cp_generator_noncausal.exists()
    cp = torch.load(path_cp_generator_noncausal, map_location=lambda storage, loc: storage)
    generator_noncausal.load_state_dict(cp)
    del cp
    print(f"loaded {path_cp_generator_noncausal}")

    style_encoder_noncausal = StyleEncoder(num_domains=num_speaker, style_dim=h.style_dim).to(device)
    assert path_cp_style_encoder_noncausal.exists()
    cp = torch.load(path_cp_style_encoder_noncausal, map_location=lambda storage, loc: storage)
    style_encoder_noncausal.load_state_dict(cp)
    del cp
    print(f"loaded {path_cp_style_encoder_noncausal}")

    recognizer_f0.eval_grad()
    recognizer_ppg.eval()

    # prepare loss function

    f_bceloss = BCEWithLogitHingeLoss()
    f_celoss = CrossEntropyLoss()

    # prepare tensorboard

    sw = SummaryWriter(path_dir_log)

    list_path_sample = load_list(path_dir_list / "valid_sample.txt")
    list_index_sample = [dataset_valid.list_path_wav_input.index(path_sample) for path_sample in list_path_sample]
    len_list_first_sample = len(list_index_sample)
    count_sample = 0

    list_index_sample = [index + dataset_valid.len_list_input * i for index in list_index_sample for i in range(len(dataset_valid.list_num_target))]
    list_index_sample = [(i // h.batch_size_valid, i % h.batch_size_valid) for i in list_index_sample]
    list_path_sample = [path for path in list_path_sample for i in range(len(dataset_valid.list_num_target))]
    len_list_sample = len(list_index_sample)

    plot_melspe = MelSpectrogramPlotter(hv)

    # start training

    while(epoch < h.epochs):
        epoch += 1
        print(f"--- epoch {epoch} train ---")

        # train

        for batch in tqdm(dataloader_train):
            step += 1

            spe_input, num_input, spe_target, num_target = [item.to(device) for item in batch]
            spe_input_cut = spe_input[:, :, :, generator.tail:]

            # forward discriminator

            [item.train() for item in [discriminator, classifier]]

            with torch.no_grad():
                style_target = style_encoder(spe_target, num_target)
                spe_output = generator(spe_input, style_target)

            spe_input_cut.requires_grad_()
            r_input = discriminator(spe_input_cut, num_input)
            r_output = discriminator(spe_output, num_target)

            loss_d_real = f_bceloss(r_input, 1)
            loss_d_fake = f_bceloss(r_output, 0)
            r1reg_d = f_r1reg(r_input, spe_input_cut)
            loss_d = loss_d_real + loss_d_fake + h.lambda_r1reg * r1reg_d

            # backward discriminator

            optimizer_discriminator.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_discriminator.step()

            if epoch >= h.epoch_start_classify:
                # forward classifier

                p_input = classifier(spe_input_cut)
                p_output = classifier(spe_output)

                loss_c_real = f_celoss(p_input, num_input)
                loss_c_fake = f_celoss(p_output, num_input)
                r1reg_c = f_r1reg(p_input, spe_input_cut)
                loss_c = loss_c_real + loss_c_fake + h.lambda_r1reg * r1reg_c

                # backward classifier

                optimizer_classifier.zero_grad(set_to_none=True)
                loss_c.backward()
                optimizer_classifier.step()

            else:
                loss_c_real = 0.0
                loss_c_fake = 0.0

            # forward generator, style_encoder

            [item.eval() for item in [discriminator, classifier]]

            style_target = style_encoder(spe_target, num_target)
            spe_output = generator(spe_input, style_target)

            r_output = discriminator(spe_output, num_target)
            loss_g_adv = f_bceloss(r_output, 1)

            if epoch >= h.epoch_start_classify:
                p_output = classifier(spe_output)
                loss_g_advcls = f_celoss(p_output, num_target)
            else:
                loss_g_advcls = 0.0

            with torch.no_grad():
                style_input = style_encoder_noncausal(spe_input_cut, num_input)

            spe_input_cycle = generator_noncausal(spe_output, style_input)
            loss_cycle = F.smooth_l1_loss(spe_input_cycle, spe_input_cut)

            with torch.no_grad():
                f0_input, vuv_input = recognizer_f0(standardizer_recognizer_f0(spe_input_cut))
                vuv_input = vuv_input >= 0.5
                ppg_input = recognizer_ppg(standardizer_recognizer_ppg(spe_input_cut))

            f0_output, vuv_output = recognizer_f0(standardizer_recognizer_f0(spe_output))
            loss_f0 = f_loss_f0(f0_output, vuv_output.detach() >= 0.5, f0_input, vuv_input)

            ppg_output = recognizer_ppg(standardizer_recognizer_ppg(spe_output))
            loss_ppg = F.smooth_l1_loss(ppg_output, ppg_input)

            loss_g = h.lambda_adv * loss_g_adv + h.lambda_advcls * loss_g_advcls + h.lambda_cycle * loss_cycle + h.lambda_f0 * loss_f0 + h.lambda_ppg * loss_ppg

            # backward generator, style_encoder

            optimizer_style_encoder.zero_grad(set_to_none=True)
            optimizer_generator.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_style_encoder.step()
            optimizer_generator.step()

            # write log

            sw.add_scalar("train/loss_d_adv_real", loss_d_real, step)
            sw.add_scalar("train/loss_d_adv_fake", loss_d_fake, step)
            sw.add_scalar("train/loss_c_advcls_real", loss_c_real, step)
            sw.add_scalar("train/loss_c_advcls_fake", loss_c_fake, step)
            sw.add_scalar("train/loss_g_adv", loss_g_adv, step)
            sw.add_scalar("train/loss_g_advcls", loss_g_advcls, step)
            sw.add_scalar("train/loss_cycle", loss_cycle, step)
            sw.add_scalar("train/loss_f0", loss_f0, step)
            sw.add_scalar("train/loss_ppg", loss_ppg, step)

        # valid

        if epoch % h.valid_interval == 0:
            print(f"--- epoch {epoch} valid ---")

            with torch.no_grad():
                loss_cycle = 0.0
                loss_f0 = 0.0
                loss_ppg = 0.0
                i = 0
                for j, batch in tqdm(enumerate(dataloader_valid)):
                    # forward

                    spe_input, num_input, spe_target, num_target = [item.to(device) for item in batch]
                    spe_input_cut = spe_input[:, :, :, generator.tail:]

                    style_target = style_encoder(spe_target, num_target)
                    spe_output = generator(spe_input, style_target)

                    style_input = style_encoder(spe_input_cut, num_input)
                    spe_input_cycle = generator_noncausal(spe_output, style_input)
                    loss_cycle += F.smooth_l1_loss(spe_input_cycle, spe_input_cut)

                    f0_input, vuv_input = recognizer_f0(standardizer_recognizer_f0(spe_input_cut))
                    f0_output, vuv_output = recognizer_f0(standardizer_recognizer_f0(spe_output))
                    loss_f0 += f_loss_f0(f0_output, vuv_output.detach() >= 0.5, f0_input, vuv_input.detach() >= 0.5)

                    ppg_input = recognizer_ppg(standardizer_recognizer_ppg(spe_input_cut))
                    ppg_output = recognizer_ppg(standardizer_recognizer_ppg(spe_output))
                    loss_ppg += F.smooth_l1_loss(ppg_output, ppg_input)

                    if i < len_list_sample:
                        if j == list_index_sample[i][0]:
                            if count_sample < len_list_first_sample:
                                name_sample_input = f"{list_path_sample[i].parent.stem}_{list_path_sample[i].stem}"
                                spe_input_j = destandardizer(spe_input_cut[list_index_sample[i][1]].squeeze(0))
                                sw.add_figure('input/spe_{}'.format(name_sample_input),
                                              plot_melspe(spe_input_j.cpu().numpy()), 0)
                                count_sample += 1
                            name_target = dict_speaker[num_target[list_index_sample[i][1]].item()]
                            name_sample_output = f"{list_path_sample[i].parent.stem}_{list_path_sample[i].stem}_to_{name_target}"
                            spe_output_j = destandardizer(spe_output[list_index_sample[i][1]].squeeze(0))
                            sw.add_figure('output/spe_{}'.format(name_sample_output),
                                          plot_melspe(spe_output_j.cpu().numpy()), step)
                            i += 1

                loss_cycle /= len(dataloader_valid)
                loss_f0 /= len(dataloader_valid)
                loss_ppg /= len(dataloader_valid)

                # write log
                sw.add_scalar("valid/loss_cycle", loss_cycle, step)
                sw.add_scalar("valid/loss_f0", loss_f0, step)
                sw.add_scalar("valid/loss_ppg", loss_ppg, step)

                # save state

                torch.save(
                    {"generator": generator.state_dict(),
                     "style_encoder": style_encoder.state_dict(),
                     "discriminator": discriminator.state_dict(),
                     "classifier": classifier.state_dict(),
                     "optimizer_generator": optimizer_generator.state_dict(),
                     "optimizer_style_encoder": optimizer_style_encoder.state_dict(),
                     "optimizer_discriminator": optimizer_discriminator.state_dict(),
                     "optimizer_classifier": optimizer_classifier.state_dict(),
                     "epoch": epoch,
                     "step": step},
                    path_dir_checkpoint / "state_latest.cp")

                print("saved state_latest.cp")

        if epoch % h.save_interval == 0 and epoch >= h.save_start:
            torch.save(
                generator.state_dict(),
                path_dir_checkpoint / f"generator_{str(epoch).zfill(4)}.cp")
            torch.save(
                style_encoder.state_dict(),
                path_dir_checkpoint / f"style_encoder_{str(epoch).zfill(4)}.cp")

            print(f"saved generator_{str(epoch).zfill(4)}.cp and style_encoder_{str(epoch).zfill(4)}.cp")
