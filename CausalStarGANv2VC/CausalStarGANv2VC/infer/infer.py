import sys
from importlib import import_module
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from tqdm import tqdm
import torch

from .dataset import SpeValidDataset
from ..tools.destandardizer import Destandardizer
from ..models.generator import Generator, StyleEncoder
from ..tools.speaker_dict import get_dict_speaker

from CausalHiFiGAN.tools.file_io import load_json, load_list
from CausalHiFiGAN.tools.wav_io import write_wav
from CausalHiFiGAN.tools.plot import MelSpectrogramPlotter
from CausalHiFiGAN.models.generator import Generator as Generator_HiFiGAN

torch.backends.cudnn.benchmark = True


def infer(path_dir_list=Path("./data/list"),
          path_dir_param=Path("./data/param"),
          path_dir_checkpoint=Path("./checkpoint"),
          path_result=Path("./result"),
          path_config="./config.json",
          path_config_vocoder="../CausalHiFiGAN/config_v1.json",
          path_CausalHiFiGAN="../CausalHiFiGAN"):
    print("--- infer ---")

    # prepare directory

    path_checkpoint_CausalHiFiGAN = Path(f"{path_CausalHiFiGAN}/checkpoint/g_01000000")
    path_result.mkdir(exist_ok=1)

    # load config

    h = load_json(path_config)
    hv = load_json(path_config_vocoder)

    device = torch.device(h.device)

    # prepare model

    dict_speaker = get_dict_speaker(path_dir_list / "train")
    num_speaker = len(dict_speaker)

    converter = Generator(style_dim=h.style_dim).to(device)
    style_encoder = StyleEncoder(num_domains=num_speaker, style_dim=h.style_dim).to(device)
    vocoder = Generator_HiFiGAN(hv).to(device)

    assert path_checkpoint_CausalHiFiGAN.exists()
    cp = torch.load(path_checkpoint_CausalHiFiGAN, map_location=lambda storage, loc: storage)
    vocoder.load_state_dict(cp["generator"])
    print(f"loaded {path_checkpoint_CausalHiFiGAN}")

    [item.eval() for item in [converter, style_encoder, vocoder]]
    vocoder.remove_weight_norm()

    # prepare dataset

    dataset = SpeValidDataset(path_dir_list / "valid",
                              path_dir_list / "valid_target",
                              path_dir_param / "stats.json",
                              dict_speaker,
                              hv,
                              converter.tail * hv.hop_size)

    destandardizer = Destandardizer(path_dir_param / "stats.json")

    # prepare plotter

    plot_melspe = MelSpectrogramPlotter(hv)

    # infer for each model

    list_path_cp_generator = sorted(path_dir_checkpoint.glob("generator_????.cp"))
    list_path_cp_style_encoder = sorted(path_dir_checkpoint.glob("style_encoder_????.cp"))

    for path_cp_generator, path_cp_style_encoder in zip(list_path_cp_generator, list_path_cp_style_encoder):
        epoch = path_cp_generator.name[10:14]
        print(f" -- inference with epoch {epoch} models --")

        path_out_epoch = path_result / epoch
        path_out_epoch.mkdir(exist_ok=1)
        path_out_wav = path_out_epoch / "wav"
        path_out_wav.mkdir(exist_ok=1)
        path_out_mel = path_out_epoch / "melspectrogram"
        path_out_mel.mkdir(exist_ok=1)

        cp = torch.load(path_cp_generator, map_location=lambda storage, loc: storage)
        converter.load_state_dict(cp)
        print(f"loaded {path_cp_generator}")
        cp = torch.load(path_cp_style_encoder, map_location=lambda storage, loc: storage)
        style_encoder.load_state_dict(cp)
        print(f"loaded {path_cp_style_encoder}")
        del cp
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataset)):
                spe_input, spe_target, num_target = [item.to(device) for item in batch]

                style_target = style_encoder(spe_target, num_target)
                spe_output = converter(spe_input, style_target)
                spe_output = destandardizer(spe_output.squeeze(1))
                wav_output = vocoder(spe_output)

                name_target = dict_speaker[num_target[0].item()]
                path_wav_input = dataset.list_path_wav_input[i % dataset.len_list_input]
                name_output = f"{path_wav_input.parent.stem}_{path_wav_input.stem}_to_{name_target}"
                write_wav(path_out_wav / f"{name_output}.wav", wav_output, hv.sampling_rate)

                fig = plot_melspe(spe_output.squeeze().cpu().numpy())
                fig.savefig(path_out_mel / f"{name_output}.png")
