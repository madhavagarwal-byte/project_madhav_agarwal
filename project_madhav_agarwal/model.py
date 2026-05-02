import torch
import torch.nn as nn

from config import INPUT_CHANNELS, RESIZE_X, RESIZE_Y


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )


class IDCClassifier(nn.Module):
    """
    Lightweight 3-block CNN for IDC binary classification.

    Architecture:
        (B, 3, 50, 50)
        Block1: Conv(3→32)   BN ReLU MaxPool → (B, 32, 25, 25)
        Block2: Conv(32→64)  BN ReLU MaxPool → (B, 64, 12, 12)
        Block3: Conv(64→128) BN ReLU MaxPool → (B, 128, 6, 6)
        Flatten → 4608
        FC(4608→256) BN ReLU Dropout(0.5)
        FC(256→1)  raw logit
    """

    def __init__(self, in_channels=INPUT_CHANNELS,
                 img_h=RESIZE_Y, img_w=RESIZE_X):
        super().__init__()
        self.block1 = _conv_block(in_channels, 32)
        self.block2 = _conv_block(32, 64)
        self.block3 = _conv_block(64, 128)
        self._flat  = self._get_flat(in_channels, img_h, img_w)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _get_flat(self, c, h, w):
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = self.block3(self.block2(self.block1(x)))
            return x.numel()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)          # (B, 1) raw logit

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))   # (B, 1) in [0, 1]

    def predict(self, x):
        return (self.predict_proba(x) > 0.5).int()  # (B, 1) in {0, 1}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = IDCClassifier()
    x = torch.randn(4, INPUT_CHANNELS, RESIZE_Y, RESIZE_X)
    print(f"Output shape : {m(x).shape}")
    print(f"Parameters   : {m.count_parameters():,}")
