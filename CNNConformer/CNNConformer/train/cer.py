from torchaudio.functional import edit_distance

from ..tools.ctc import CTC, CTC_label


class CER:
    def __init__(self, blank=0):
        self.ctc_prob = CTC(blank)
        self.ctc_lab = CTC_label(blank)

    def __call__(self, p_lab_h, lab):
        """
        p_lab_h: FloatTensor[batch,sequence,class]
        lab: LongTensor[batch,sequence]
        """
        cer = 0.0
        for p_lab_h_i, lab_i in zip(p_lab_h, lab):
            lab_h_i = self.ctc_prob(p_lab_h_i)
            lab_i = self.ctc_lab(lab_i)
            cer += edit_distance(lab_h_i, lab_i) / lab_i.size(-1)
        cer /= lab.size(0)
        return cer
