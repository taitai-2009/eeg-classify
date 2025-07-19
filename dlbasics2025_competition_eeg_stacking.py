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
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from einops import repeat
from glob import glob
from termcolor import cprint
#from tqdm.notebook import tqdm
from tqdm import tqdm

import torchvision.models as models
from torch.optim.lr_scheduler import OneCycleLR

from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Additional imports for stacking
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

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
            # Variable Gaussian noise: sigma sampled from [0.005, 0.02]
            sigma = random.uniform(0.005, 0.02)
            X = X + torch.randn_like(X) * sigma
            # Random gain scaling: factor from [0.9, 1.1]
            gain = random.uniform(0.9, 1.1)
            X = X * gain
            # Channel flip with 50% chance
            if random.random() < 0.5:
                X = X.flip(dims=[0])
            # Random channel drop: zero out one channel with 10% chance
            if random.random() < 0.1:
                ch = random.randrange(X.size(0))
                X[ch] = 0
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]

## 3.モデル定義

 # Unused in final ensemble (did not yield improved score)
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


 # Unused in final ensemble (did not yield improved score)
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

 # Unused in final ensemble (did not yield improved score)
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

# --- Training function for both models ---
def accuracy(y_pred, y):
    return (y_pred.argmax(dim=-1) == y).float().mean()

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, prefix):
    writer = SummaryWriter(f"tensorboard/{prefix}")
    max_val = 0.0
    for epoch in range(epochs):
        print(f"[{prefix}] Epoch {epoch+1}/{epochs}")
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc=f"Train-{prefix}"):
            X, y = X.to("cuda"), y.to("cuda")
            # MixUp augmentation
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            batch_size = X.size(0)
            index = torch.randperm(batch_size).to("cuda")
            X2, y2 = X[index], y[index]
            X_mix = lam * X + (1 - lam) * X2
            # Forward
            y_pred = model(X_mix)
            loss = lam * F.cross_entropy(y_pred, y, label_smoothing=0.1) + \
                   (1 - lam) * F.cross_entropy(y_pred, y2, label_smoothing=0.1)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # step the OneCycleLR scheduler per batch
            scheduler.step()
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())
        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc=f"Validation-{prefix}"):
            X, y = X.to("cuda"), y.to("cuda")
            with torch.no_grad():
                y_pred = model(X)
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())
        print(f"[{prefix}] Epoch {epoch+1}/{epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        writer.add_scalar("train_loss", np.mean(train_loss), epoch)
        writer.add_scalar("train_acc", np.mean(train_acc), epoch)
        writer.add_scalar("val_loss", np.mean(val_loss), epoch)
        writer.add_scalar("val_acc", np.mean(val_acc), epoch)
        #scheduler.step(np.mean(val_acc))  # Not used for OneCycleLR
        torch.save(model.state_dict(), f"{prefix}_last.pt")
        if np.mean(val_acc) > max_val:
            cprint(f"New best for {prefix}. Saving weights.", "cyan")
            torch.save(model.state_dict(), f"{prefix}_best.pt")
            max_val = np.mean(val_acc)
    return f"{prefix}_best.pt"

# Hyperparams
epochs = 50
lr = 0.001
batch_size = 512

# DataLoaders
train_set = RawEEGDataset("train")
val_set = RawEEGDataset("val")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Train Transformer
model_t = TransformerClassifier(
    in_ch=train_set.num_channels,
    hid_dim=128,
    nhead=4,
    num_layers=2,
    num_classes=train_set.num_classes
).to("cuda")
opt_t = torch.optim.Adam(model_t.parameters(), lr=lr)
sched_t = OneCycleLR(
    opt_t,
    max_lr=lr,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy="linear"
)
path_t = train_model(model_t, train_loader, val_loader, opt_t, sched_t, epochs, "transformer")

# Train CNN-LSTM
model_c = CNN_LSTM_Classifier(
    in_ch=train_set.num_channels,
    hid_dim=128,
    lstm_hidden=64,
    lstm_layers=1,
    num_classes=train_set.num_classes
).to("cuda")
opt_c = torch.optim.Adam(model_c.parameters(), lr=lr)
sched_c = OneCycleLR(
    opt_c,
    max_lr=lr,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy="linear"
)
path_c = train_model(model_c, train_loader, val_loader, opt_c, sched_c, epochs, "cnnlstm")


# ------------------
#  Ensemble Validation Accuracy
# ------------------
val_set_ens = RawEEGDataset("val")
val_loader_ens = DataLoader(val_set_ens, batch_size=batch_size, shuffle=False)
model1 = TransformerClassifier(
    in_ch=val_set_ens.num_channels,
    hid_dim=128,
    nhead=4,
    num_layers=2,
    num_classes=val_set_ens.num_classes
).to("cuda")
model1.load_state_dict(torch.load("transformer_best.pt", map_location="cpu"))
model1.eval()
model2 = CNN_LSTM_Classifier(
    in_ch=val_set_ens.num_channels,
    hid_dim=128,
    lstm_hidden=64,
    lstm_layers=1,
    num_classes=val_set_ens.num_classes
).to("cuda")
model2.load_state_dict(torch.load("cnnlstm_best.pt", map_location="cpu"))
model2.eval()

'''
# ------------------
#  Grid search for best ensemble weights
# ------------------
best_acc, best_w1 = 0.0, 0.0
for w1 in np.linspace(0, 1, 11):
    w2 = 1 - w1
    correct, total = 0, 0
    for X, y, _ in val_loader_ens:
        X, y = X.to("cuda"), y.to("cuda")
        with torch.no_grad():
            p1 = model1(X)
            p2 = model2(X)
            p = w1 * p1 + w2 * p2
            preds = p.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc, best_w1 = acc, w1
best_w2 = 1 - best_w1
print(f"Best ensemble w1={best_w1:.2f}, w2={best_w2:.2f} -> val_acc={best_acc:.4f}")

# Fine grid search around best_w1
fine_best_acc, fine_best_w1 = best_acc, best_w1
low = max(0, best_w1 - 0.1)
high = min(1, best_w1 + 0.1)
for w1 in np.linspace(low, high, 21):
    w2 = 1 - w1
    correct, total = 0, 0
    for X, y, _ in val_loader_ens:
        X, y = X.to("cuda"), y.to("cuda")
        with torch.no_grad():
            p = w1 * model1(X) + w2 * model2(X)
            preds = p.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    if acc > fine_best_acc:
        fine_best_acc, fine_best_w1 = acc, w1
best_w1, best_acc = fine_best_w1, fine_best_acc
best_w2 = 1 - best_w1
print(f"Refined best w1={best_w1:.3f}, w2={best_w2:.3f} -> val_acc={best_acc:.4f}")
'''

'''
# ------------------
#  Ensemble Val with best weights
# ------------------
correct, total = 0, 0
for X, y, _ in tqdm(val_loader_ens, desc="Ensemble Val with Best Weights"):
    with torch.no_grad():
        p1 = model1(X)
        p2 = model2(X)
        p = best_w1 * p1 + best_w2 * p2
        preds = p.argmax(dim=-1)
    correct += (preds == y).sum().item()
    total += y.size(0)
val_acc_ens = correct / total
print(f"Ensemble Validation Top-1 Accuracy with best weights: {val_acc_ens:.4f}")
'''

# ------------------
#  7. Stacking with Meta-Models
# ------------------
print("Collecting validation set predictions for stacking...")
probs1, probs2, ys = [], [], []
for X, y, _ in val_loader_ens:
    X = X.to("cuda")
    with torch.no_grad():
        p1 = torch.softmax(model1(X), dim=1).cpu().numpy()
        p2 = torch.softmax(model2(X), dim=1).cpu().numpy()
    probs1.append(p1)
    probs2.append(p2)
    ys.append(y.numpy())
probs1 = np.vstack(probs1)
probs2 = np.vstack(probs2)
ys     = np.concatenate(ys)
X_meta = np.hstack([probs1, probs2])

# Logistic Regression meta-model
#print("Training Logistic Regression meta-model...")
#meta_lr = LogisticRegression(max_iter=200)
#meta_lr.fit(X_meta, ys)
#y_pred_lr = meta_lr.predict(X_meta)
#acc_lr = accuracy_score(ys, y_pred_lr)
#print(f"Stacking Val Acc (LogisticRegression): {acc_lr:.4f}")

# LightGBM meta-model
print("Training LightGBM meta-model...")
meta_lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1)
meta_lgb.fit(X_meta, ys)
y_pred_lgb = meta_lgb.predict(X_meta)
acc_lgb = accuracy_score(ys, y_pred_lgb)
print(f"Stacking Val Acc (LightGBM): {acc_lgb:.4f}")



## 5.評価

# ------------------
#    Dataloader
# ------------------
test_set = RawEEGDataset("test")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model1 = TransformerClassifier(
    in_ch=test_set.num_channels, hid_dim=128, nhead=4, num_layers=2, num_classes=test_set.num_classes
).to("cuda")
model1.load_state_dict(torch.load("transformer_best.pt", map_location="cpu"))
model1.eval()

model2 = CNN_LSTM_Classifier(
    in_ch=test_set.num_channels, hid_dim=128, lstm_hidden=64, lstm_layers=1, num_classes=test_set.num_classes
).to("cuda")
model2.load_state_dict(torch.load("cnnlstm_best.pt", map_location="cpu"))
model2.eval()

# ------------------
#  8. Stacking Inference
# ------------------
print("Collecting test set predictions for stacking inference...")
probs1_t, probs2_t = [], []
for X, subject_idxs in tqdm(test_loader, desc="Stacking Inference"):
    X = X.to("cuda")
    with torch.no_grad():
        p1 = torch.softmax(model1(X), dim=1).cpu().numpy()
        p2 = torch.softmax(model2(X), dim=1).cpu().numpy()
    probs1_t.append(p1)
    probs2_t.append(p2)
probs1_t = np.vstack(probs1_t)
probs2_t = np.vstack(probs2_t)
X_meta_t = np.hstack([probs1_t, probs2_t])

# Choose meta-model for submission
final_probs = meta_lgb.predict_proba(X_meta_t)
np.save("submission", final_probs)
print(f"Stacking submission saved with shape {final_probs.shape}")

'''
# ------------------
#  6. Pseudo-Labeling Self-Training
# ------------------
print("Loading pseudo labels and performing fine-tuning...")
# Load test EEG data and pseudo labels
pseudo_preds = np.load("submission.npy")
pseudo_labels = pseudo_preds.argmax(axis=1)
pseudo_X = np.load("data/test/eeg.npy")
pseudo_X = torch.from_numpy(pseudo_X).to(torch.float32)

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        # Return subject_idxs as a tensor for consistency
        return self.X[i], self.y[i], torch.tensor(0, dtype=torch.long)

# Create pseudo dataset and combine with original train set
pseudo_set = PseudoDataset(pseudo_X, pseudo_labels)
combined_train = torch.utils.data.ConcatDataset([train_set, pseudo_set])
combined_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)

# Fine-tune both models for a few epochs
ft_epochs = 5
for prefix, model, opt, sched in [
    ("transformer_ft", model_t, opt_t, sched_t),
    ("cnnlstm_ft", model_c, opt_c, sched_c),
]:
    print(f"Fine-tuning {prefix} for {ft_epochs} epochs")
    train_model(model, combined_loader, val_loader, opt, sched, ft_epochs, prefix)
'''

## 提出方法

### 以下の3点をzip化し，Omnicampusの「最終課題 (EEG)」から提出してください．

### - `submission.npy`
### - `model_last.pt`や`model_best.pt`など，テストに使用した重み（拡張子は`.pt`のみ）
### - 本Colab Notebook


from zipfile import ZipFile

notebook_path = "dlbasics2025_competition_eeg_stacking.py"
# include both model weights separately
model_paths = ["transformer_best.pt", "cnnlstm_best.pt"]
with ZipFile("submission.zip", "w") as zf:
    zf.write("submission.npy")
    for model_path in model_paths:
        zf.write(model_path)
    zf.write(notebook_path)
