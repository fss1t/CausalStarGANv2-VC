from CausalHiFiGAN.tools.file_io import load_json


class Standardizer:
    def __init__(self, path_norm_in, path_norm_out):
        # load

        stats = load_json(path_norm_in)
        self.mean_in = stats["mean"]
        self.std_in = stats["std"]

        stats = load_json(path_norm_out)
        self.mean_out = stats["mean"]
        self.std_out = stats["std"]

    def __call__(self, spe):
        return (spe * self.std_in + self.mean_in - self.mean_out) / self.std_out
