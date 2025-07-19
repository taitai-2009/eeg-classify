# -*- coding: utf-8 -*-
"""
# Deep Learning 基礎講座　最終課題: 脳波分類

## 概要
被験者が画像を見ているときの脳波から，その画像がどのカテゴリに属するかを分類するタスク．
- サンプル数: 訓練 118,800 サンプル，検証 59,400 サンプル，テスト 59,400 サンプル
- クラス数: 5
- 入力: 脳波データ（チャンネル数 x 系列長）
- 出力: 対応する画像のクラス
- 評価指標: Top-1 accuracy

### 元データセット ([Gifford2022 EEG dataset](https://osf.io/3jk45/)) との違い

- 本コンペでは難易度調整の目的で元データセットにいくつかの改変を加えています．

1. 訓練セットのみの使用
  - 元データセットでは訓練データに存在しなかったクラスの画像を見ているときの脳波においてテストが行われますが，これは難易度が非常に高くなります．
  - 本コンペでは元データセットの訓練セットを再分割し，訓練時に存在した画像に対応する別の脳波において検証・テストを行います．

2. クラス数の減少
  - 元データセット（の訓練セット）では16,540枚の画像に対し，1,654のクラスが存在します．
    - e.g. `aardvark`, `alligator`, `almond`, ...
  - 本コンペでは1,654のクラスを，`animal`, `food`, `clothing`, `tool`, `vehicle`の5つにまとめています．
    - e.g. `aardvark -> animal`, `alligator -> animal`, `almond -> food`, ...

### 考えられる工夫の例

- 音声モデルの導入
  - 脳波と同じ波である音声を扱うアーキテクチャを用いることが有効であると知られています．
  - 例）Conformer [[Gulati+ 2020](https://arxiv.org/abs/2005.08100)]
- 画像データを用いた事前学習
  - 本コンペのタスクは脳波のクラス分類ですが，配布してある画像データを脳波エンコーダの事前学習に用いることを許可します．
  - 例）CLIP [Radford+ 2021]
  - 画像を用いる場合は[こちら](https://osf.io/download/3v527/)からダウンロードしてください．
- 過学習を防ぐ正則化やドロップアウト

## 修了要件を満たす条件
- ベースラインモデルのbest test accuracyは38.7%となります．**これを超えた提出のみ，修了要件として認めます**．
- ベースラインから改善を加えることで，55%までは性能向上することを運営で確認しています．こちらを 1 つの指標として取り組んでみてください．

## 注意点
- 学習するモデルについて制限はありませんが，必ず訓練データで学習したモデルで予測してください．
    - 事前学習済みモデルを利用して，訓練データを fine-tuning しても構いません．
    - 埋め込み抽出モデルなど，モデルの一部を訓練しないケースは構いません．
    - 学習を一切せずに，ChatGPT などの基盤モデルを利用することは禁止とします．

"""
## 1.準備


# omnicampus 実行用
#!pip install ipywidgets

# ライブラリのインポートとシード固定
import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange
from einops import repeat
from glob import glob
from termcolor import cprint
#from tqdm.notebook import tqdm
from tqdm import tqdm

import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn import TransformerEncoder, TransformerEncoderLayer

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ドライブのマウント（Colabの場合）
#from google.colab import drive
#drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# ワーキングディレクトリを作成し移動．ノートブックを配置したディレクトリに適宜書き換え
#WORK_DIR = "/content/drive/MyDrive/weblab/DLBasics2025/Competition"
#WORK_DIR = "/content/drive/MyDrive/Colab Notebooks/2025DL基礎/3_最終課題/EEG"
#os.makedirs(WORK_DIR, exist_ok=True)
# %cd {WORK_DIR}

## 2.データセット
### ノートブックと同じディレクトリに`data/`が存在することを確認してください．


class ThingsEEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 5
        self.num_subjects = 10

        self.X = np.load(f"data/{split}/eeg.npy")
        self.X = torch.from_numpy(self.X).to(torch.float32)
        self.subject_idxs = np.load(f"data/{split}/subject_idxs.npy")
        self.subject_idxs = torch.from_numpy(self.subject_idxs)

        if split in ["train", "val"]:
            self.y = np.load(f"data/{split}/labels.npy")
            self.y = torch.from_numpy(self.y)

        print(f"EEG: {self.X.shape}, labels: {self.y.shape if hasattr(self, 'y') else None}, subject indices: {self.subject_idxs.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]  # (C, T)
        specs = []
        for c in range(X.shape[0]):
            # compute STFT: returns [freq_bins, time_steps, 2]
            spec_c = torch.stft(
                X[c],
                n_fft=32,
                win_length=32,
                hop_length=16,
                return_complex=False,
                center=True,
            )  # shape (F, T', 2)
            # magnitude squared
            spec_c = spec_c.pow(2).sum(-1)  # (F, T')
            specs.append(spec_c.unsqueeze(0))  # (1, F, T')
        X = torch.cat(specs, dim=0)  # (C, F, T')
        # reshape for 2D-CNN
        X = X.unsqueeze(0).reshape(1, -1, X.shape[2])
        if self.split == "train":
            noise = torch.randn_like(X) * 0.01  # adjust scale as needed
            X = X + noise
            # channel flip with 50% chance
            if random.random() < 0.5:
                X = X.flip(dims=[1])
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

class RawEEGDataset(ThingsEEGDataset):
    """Return raw EEG time-series (C, T) for CNN/LSTM models."""
    def __getitem__(self, i):
        X = self.X[i]  # (C, T)
        if self.split == "train":
            # Gaussian noise
            X = X + torch.randn_like(X) * 0.01
            # channel flip
            if random.random() < 0.5:
                X = X.flip(dims=[0])
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]

## 3.ベースラインモデル

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_sizes=(3,5,7), p_drop=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, k, padding="same") for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(p_drop)
    def forward(self, X):
        outs = [conv(X) for conv in self.convs]
        X = sum(outs) / len(outs)
        X = F.gelu(self.bn(X))
        return self.dropout(X)

class EEGNetBlock(nn.Module):
    def __init__(self, in_ch, depth_ch, point_ch, kernel_len, p_drop=0.5):
        super().__init__()
        # 1. Temporal convolution
        self.temp_conv = nn.Conv1d(in_ch, depth_ch, kernel_size=kernel_len,
                                   padding=kernel_len//2, bias=False)
        self.bn1 = nn.BatchNorm1d(depth_ch)
        # 2. Depthwise spatial conv
        self.depth_conv = nn.Conv1d(depth_ch, depth_ch, kernel_size=1,
                                    groups=depth_ch, bias=False)
        self.bn2 = nn.BatchNorm1d(depth_ch)
        # 3. Pointwise convolution
        self.point_conv = nn.Conv1d(depth_ch, point_ch, kernel_size=1,
                                    bias=False)
        self.bn3 = nn.BatchNorm1d(point_ch)
        self.dropout = nn.Dropout(p_drop)
    def forward(self, x):
        x = F.elu(self.bn1(self.temp_conv(x)))
        x = F.elu(self.bn2(self.depth_conv(x)))
        x = F.elu(self.bn3(self.point_conv(x)))
        return self.dropout(x)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            EEGNetBlock(in_channels, depth_ch=32, point_ch=hid_dim, kernel_len=64, p_drop=0.5),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)

class EEGSpecClassifier(nn.Module):
    def __init__(self, num_classes, in_ch=1):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=True)
        # Adjust first conv to accept in_ch channels
        self.backbone.features[0][0] = nn.Conv2d(
            in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Replace classifier head
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, num_classes
        )
    def forward(self, x):
        # x shape: (B, in_ch, H, W)
        return self.backbone(x)

class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, in_ch, hid_dim, lstm_hidden, lstm_layers, num_classes):
        super().__init__()
        # 1D-CNN feature extractor
        self.cnn = EEGNetBlock(in_ch, depth_ch=32, point_ch=hid_dim, kernel_len=64, p_drop=0.5)
        # fix time dimension T to 10 via adaptive pooling
        self.pool = nn.AdaptiveAvgPool1d(10)
        # bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hid_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        # classifier
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
    def forward(self, x):
        # x: (B, C, T)
        x = self.cnn(x)          # (B, hid_dim, T)
        x = self.pool(x)         # (B, hid_dim, 10)
        x = x.permute(0, 2, 1)   # (B, 10, hid_dim)
        o, _ = self.lstm(x)      # (B, 10, 2*lstm_hidden)
        o = o.mean(dim=1)        # (B, 2*lstm_hidden)
        return self.classifier(o)

class TransformerClassifier(nn.Module):
    def __init__(self, in_ch, hid_dim, nhead, num_layers, num_classes):
        super().__init__()
        # 1D-CNN preprocessing
        self.cnn = EEGNetBlock(in_ch, depth_ch=32, point_ch=hid_dim, kernel_len=64, p_drop=0.5)
        # fix time dimension T to 10
        self.pool = nn.AdaptiveAvgPool1d(10)
        # positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 10, hid_dim))
        # transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # classifier
        self.classifier = nn.Linear(hid_dim, num_classes)
    def forward(self, x):
        # x: (B, C, T)
        x = self.cnn(x)             # (B, hid_dim, T)
        x = self.pool(x)            # (B, hid_dim, 10)
        x = x.permute(0, 2, 1)      # (B, 10, hid_dim)
        x = x + self.pos_enc        # add positional encoding
        x = self.transformer(x)     # (B, 10, hid_dim)
        x = x.mean(dim=1)           # (B, hid_dim)
        return self.classifier(x)

## 4.訓練実行

# ハイパラ
lr = 0.001
batch_size = 512
#epochs = 80
epochs = 50

# ------------------
#    Dataloader
# ------------------
train_set = RawEEGDataset("train") # ThingsMEGDataset("train")
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
val_set = RawEEGDataset("val") # ThingsMEGDataset("val")
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False
)

# ------------------
#       Model
# ------------------
model = TransformerClassifier(
    in_ch=train_set.num_channels,
    hid_dim=128,
    nhead=4,
    num_layers=2,
    num_classes=train_set.num_classes
).to("cuda")

# ------------------
#     Optimizer
# ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    threshold=1e-4
)

# ------------------
#   Start training
# ------------------
max_val_acc = 0
def accuracy(y_pred, y):
    return (y_pred.argmax(dim=-1) == y).float().mean()

writer = SummaryWriter("tensorboard")

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    model.train()
    for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
        X, y = X.to("cuda"), y.to("cuda")

        y_pred = model(X)

        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(y_pred, y)
        train_acc.append(acc.item())

    model.eval()
    for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
        X, y = X.to("cuda"), y.to("cuda")

        with torch.no_grad():
            y_pred = model(X)

        val_loss.append(F.cross_entropy(y_pred, y).item())
        val_acc.append(accuracy(y_pred, y).item())

    print(f"Epoch {epoch+1}/{epochs} | \
        train loss: {np.mean(train_loss):.3f} | \
        train acc: {np.mean(train_acc):.3f} | \
        val loss: {np.mean(val_loss):.3f} | \
        val acc: {np.mean(val_acc):.3f}")

    writer.add_scalar("train_loss", np.mean(train_loss), epoch)
    writer.add_scalar("train_acc", np.mean(train_acc), epoch)
    writer.add_scalar("val_loss", np.mean(val_loss), epoch)
    writer.add_scalar("val_acc", np.mean(val_acc), epoch)

    # adjust learning rate based on validation accuracy
    scheduler.step(np.mean(val_acc))

    torch.save(model.state_dict(), "model_last.pt")

    if np.mean(val_acc) > max_val_acc:
        cprint("New best. Saving the model.", "cyan")
        torch.save(model.state_dict(), "model_best.pt")
        max_val_acc = np.mean(val_acc)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir tensorboard

## 5.評価

# ------------------
#    Dataloader
# ------------------
test_set = RawEEGDataset("test")
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)

# ------------------
#       Model
# ------------------
model = TransformerClassifier(
    in_ch=test_set.num_channels,
    hid_dim=128,
    nhead=4,
    num_layers=2,
    num_classes=test_set.num_classes
).to("cuda")
model.load_state_dict(torch.load("model_best.pt", map_location="cuda"))

# ------------------
#  Start evaluation
# ------------------
preds = []
model.eval()
for X, subject_idxs in tqdm(test_loader, desc="Evaluation"):
    preds.append(model(X.to("cuda")).detach().cpu())

preds = torch.cat(preds, dim=0).numpy()
np.save("submission", preds)
print(f"Submission {preds.shape} saved.")

## 提出方法

### 以下の3点をzip化し，Omnicampusの「最終課題 (EEG)」から提出してください．

### - `submission.npy`
### - `model_last.pt`や`model_best.pt`など，テストに使用した重み（拡張子は`.pt`のみ）
### - 本Colab Notebook


from zipfile import ZipFile

model_path = "model_best.pt"
notebook_path = "dlbasics2025_competition_eeg_transformer.py"

with ZipFile("submission.zip", "w") as zf:
    zf.write("submission.npy")
    zf.write(model_path)
    zf.write(notebook_path)
