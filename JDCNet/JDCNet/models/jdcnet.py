"""
Implementation of model from:

Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)

Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import torch
from torch import nn


class JDCNet(nn.Module):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """

    def __init__(self, leaky_relu_slope=0.01):
        super().__init__()

        # input = (b, 1, 192, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),  # out: (b, 64, 192, 513)
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  # (b, 64, 192, 513)
        )

        # res blocks
        self.res_block1 = ResBlock(in_channels=64, out_channels=128)  # (b, 128, 192, 128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)  # (b, 192, 192, 32)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)  # (b, 256, 192, 8)

        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  # (b, 256, 192, 2)
            nn.Dropout(p=0.5),
        )

        # maxpool layers (for auxiliary network inputs)
        # in = (b, 128, 192, 513) from conv_block, out = (b, 128, 192, 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        # in = (b, 128, 192, 128) from res_block1, out = (b, 128, 192, 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        # in = (b, 128, 192, 32) from res_block2, out = (b, 128, 192, 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))

        # in = (b, 640, 192, 2), out = (b, 256, 192, 2)
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.5),
        )

        # input: (b, 192, 512) - resized from (b, 256, 192, 2)
        self.bilstm_classifier = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, dropout=0.3, bidirectional=True)  # (b, 192, 512)

        # input: (b, 192, 512) - resized from (b, 256, 192, 2)
        self.bilstm_detector = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, dropout=0.3, bidirectional=True)  # (b, 192, 512)

        # input: (b * 192, 512)
        self.classifier = nn.Linear(in_features=512, out_features=1)  # (b * 192)

        # input: (b * 192, 512)
        self.detector = nn.Linear(in_features=512, out_features=1)  # (b * 192) - binary classifier

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        """
        Returns:
            classification_prediction, detection_prediction
            sizes: (b, 192), (b, 192)
        """
        ###############################
        # forward pass for classifier #
        ###############################
        Nbatch = x.size(0)
        x = x.transpose(-1, -2)

        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block(resblock3_out)

        # (b, 256, 192, 2) => (b, 192, 256, 2) => (b, 192, 512)
        classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view((Nbatch, -1, 512))
        classifier_out, _ = self.bilstm_classifier(classifier_out)  # ignore the hidden states

        classifier_out = classifier_out.contiguous().view((-1, 512))  # (b * 192, 512)
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((Nbatch, -1))  # (b, 192)

        #############################
        # forward pass for detector #
        #############################
        mp1_out = self.maxpool1(convblock_out)
        mp2_out = self.maxpool2(resblock1_out)
        mp3_out = self.maxpool3(resblock2_out)

        # out = (b, 640, 192, 2)
        concat_out = torch.cat((mp1_out, mp2_out, mp3_out, poolblock_out), dim=1)
        detector_out = self.detector_conv(concat_out)

        # (b, 256, 192, 2) => (b, 192, 256, 2) => (b, 192, 512)
        detector_out = detector_out.permute(0, 2, 1, 3).contiguous().view((Nbatch, -1, 512))
        detector_out, _ = self.bilstm_detector(detector_out)  # (b, 192, 512)

        detector_out = detector_out.contiguous().view((-1, 512))
        detector_out = self.detector(detector_out)
        detector_out = detector_out.view((Nbatch, -1))  # binary classifier - (b, 192)

        # add the pitch prediction values to determine the pitch existence from the classifier network
        #pitch_pred, nonvoice_pred = torch.split(classifier_out, [self.num_class - 1, 1], dim=2)
        # sum the 'pitch class' prediction values to represent 'isvoice?'
        # classifier_detection = torch.cat(
        #    (torch.sum(pitch_pred, dim=2, keepdim=True), nonvoice_pred), dim=2)
        # add the classifier network's and detector network's values
        # detector_out = detector_out + classifier_detection  # (b, 192, 2)

        # sizes: (b, 192, 722), (b, 192, 2)
        # classifier output consists of predicted pitch classes per frame
        # detector output consists of: (isvoice, notvoice) estimates per frame
        return classifier_out, torch.sigmoid(detector_out)

    def get_feature_GAN(self, x):
        x = x.transpose(-1, -2)

        convblock_out = self.conv_block(x)

        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)

        return poolblock_out.transpose(-1, -2)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if p.data is None:
                    continue

                if len(p.shape) >= 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.normal_(p.data)

    def eval_grad(self):
        self.training = False
        for module in self.children():
            if type(module) != nn.LSTM:
                module.train(False)
        return self


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # apply downsampling on the y axis only
        )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        # 1 x 1 convolution layer to match the feature dimensions
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x


if __name__ == '__main__':
    dummy = torch.randn((10, 1, 192, 513))  # dummy random input
    jdc = JDCNet()
    clss, detect = jdc(dummy)
    pitches, nonvoice = torch.split(clss, [1, 721], dim=2)
    print(torch.sum(pitches, dim=2).size())
    print(clss.size())
    print(detect.size())
