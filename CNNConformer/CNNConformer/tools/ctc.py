import torch


class CTC:
    def __init__(self, blank):
        self.blank = blank

    def __call__(self, prob):
        """
        prob: FloatTensor[sequence,class]
        """
        lab = torch.argmax(prob, dim=-1)
        lab = torch.unique_consecutive(lab, dim=-1)
        lab = lab[lab != self.blank]
        return lab


class CTC_label:
    def __init__(self, blank):
        self.blank = blank

    def __call__(self, lab):
        """
        lab: IntTensor[sequence]
        """
        lab = torch.unique_consecutive(lab, dim=-1)
        lab = lab[lab != self.blank]
        return lab
