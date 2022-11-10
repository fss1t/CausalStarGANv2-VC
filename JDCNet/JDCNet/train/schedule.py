import math as m


class CosExp:
    def __init__(self, T1, T2, b_initial=1e-2, b_final=1e-2):
        self.T1 = T1
        self.T2 = T2
        self.bip1 = (1.0 + b_initial) / 2
        self.bim1 = (1.0 - b_initial) / 2
        self.bfp1 = (1.0 + b_final) / 2
        self.bfm1 = (1.0 - b_final) / 2
        self.bf = b_final
        self.c0 = m.pi / T1
        self.c1 = m.pi / (2.0 * (T2 - T1))

    def __call__(self, x):
        if x < self.T1:
            return (self.bip1 - self.bim1 * m.cos(self.c0 * x))
        elif x < self.T2:
            return (self.bfp1 + self.bfm1 * m.cos(self.c1 * (x - self.T1)))
        else:
            return self.bfm1 * m.exp(self.c1 * (self.T2 - x)) + self.bf
