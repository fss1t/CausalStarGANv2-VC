from CausalHiFiGAN.tools.file_io import load_json


class Destandardizer:
    def __init__(self, path_norm_in):
        # load

        stats = load_json(path_norm_in)
        self.mean_in = stats["mean"]
        self.std_in = stats["std"]

    def __call__(self, spe):
        return spe * self.std_in + self.mean_in
