from os.path import expanduser, join


import sys

import time

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset  # これはなに？
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter

from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
import librosa
import librosa.display


DATA_ROOT = "./data/basic5000"  # NIT-ATR503/"#
test_size = 0.01  # This means 480 utterances for training data
random_state = 1234


mgc_dim = 180  # メルケプストラム次数　？？
lf0_dim = 3  # 対数fo　？？ なんで次元が３？
vuv_dim = 1  # 無声or 有声フラグ　？？
bap_dim = 15  # 発話ごと非周期成分　？？

duration_linguistic_dim = 438  # question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442  # 上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim  # aoustice modelで求めたいもの

fs = 48000
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
hop_length = int(0.001 * frame_period * fs)

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

use_phone_alignment = True
acoustic_subphone_features = "coarse_coding" if use_phone_alignment else "full"  # とは？


class BinaryFileSource(FileDataSource):
    def __init__(self, data_root, dim, train):
        self.data_root = data_root
        self.dim = dim
        self.train = train

    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*.bin")))
        files = files[: len(files) - 5]  # last 5 is real testset

        train_files, test_files = train_test_split(
            files, test_size=test_size, random_state=random_state
        )
        if self.train:
            return train_files
        else:
            return test_files

    def collect_features(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, self.dim)


X = {"acoustic": {}}
Y = {"acoustic": {}}
utt_lengths = {"acoustic": {}}
for ty in ["acoustic"]:
    for phase in ["train", "test"]:
        train = phase == "train"
        x_dim = duration_linguistic_dim if ty == "duration" else acoustic_linguisic_dim
        y_dim = duration_dim if ty == "duration" else acoustic_dim
        X[ty][phase] = FileSourceDataset(
            BinaryFileSource(join(DATA_ROOT, "X_{}".format(ty)), dim=x_dim, train=train)
        )
        Y[ty][phase] = FileSourceDataset(
            BinaryFileSource(join(DATA_ROOT, "Y_{}".format(ty)), dim=y_dim, train=train)
        )
        utt_lengths[ty][phase] = np.array([len(x) for x in X[ty][phase]], dtype=np.int)


X_min = {}
X_max = {}
Y_mean = {}
Y_var = {}
Y_scale = {}

for typ in ["acoustic"]:
    X_min[typ], X_max[typ] = minmax(X[typ]["train"], utt_lengths[typ]["train"])
    Y_mean[typ], Y_var[typ] = meanvar(Y[typ]["train"], utt_lengths[typ]["train"])
    Y_scale[typ] = np.sqrt(Y_var[typ])


from torch.utils import data as data_utils


import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tnrange, tqdm
from torch import optim
import torch.nn.functional as F


class Rnn(nn.Module):
    def __init__(self, bidirectional=True, num_layers=2):
        super(Rnn, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        ##ここまでエンコーダ

        self.fc11 = nn.Linear(acoustic_linguisic_dim, acoustic_linguisic_dim)

        self.lstm2 = nn.LSTM(
            acoustic_linguisic_dim, 400, num_layers, bidirectional=bidirectional
        )
        self.fc3 = nn.Linear(self.num_direction * 400, acoustic_dim)

    def decode(self, linguistic_features):
        x = self.fc11(linguistic_features.view(linguistic_features.size()[0], 1, -1))
        x = F.relu(x)
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)

        return self.fc3(h3)  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features):

        return self.decode(linguistic_features)


# In[104]:


import pandas as pd


device = "cuda"
model = Rnn().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-3)  # 1e-3

start = time.time()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    MSE = F.mse_loss(
        recon_x.view(-1), x.view(-1,), reduction="sum"
    )  # F.binary_cross_entropy(recon_x.view(-1), x.view(-1, ), reduction='sum')
    # print('LOSS')
    # print(BCE)

    return MSE


X_acoustic_train = [
    minmax(x, X_min["train"], X_max["train"], feature_range=(0.01, 0.99))
    for x in X["acoustic"]["train"]
]
Y_acoustic_train = [
    scale(y, Y_mean["train"], Y_scale["train"]) for y in Y["acoustic"]["train"]
]
train_mora_index_lists = [
    train_mora_index_lists[i] for i in range(len(train_mora_index_lists))
]

X_acoustic_test = [
    minmax(x, X_min["train"], X_max["train"], feature_range=(0.01, 0.99))
    for x in X["acoustic"]["test"]
]
Y_acoustic_test = [
    scale(y, Y_mean["train"], Y_scale["train"]) for y in Y["acoustic"]["test"]
]
test_mora_index_lists = [
    test_mora_index_lists[i] for i in range(len(test_mora_index_lists))
]


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        tmp = []

        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).to(device))

        optimizer.zero_grad()
        recon_batch = model(tmp[0])
        loss = loss_function(recon_batch, tmp[1])
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        del tmp
        if batch_idx % 4945 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx,
                    train_num,
                    100.0 * batch_idx / train_num,
                    loss.item(),
                )
            )

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader)
        )
    )

    return train_loss / len(train_loader)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data, in enumerate(test_loader):
            tmp = []

            for j in range(2):
                tmp.append(torch.tensor(data[j]).to(device))

            recon_batch = model(tmp[0])
            test_loss += loss_function(recon_batch, tmp[1]).item()

            del tmp

    test_loss /= len(test_loader)
    print("====> Test set loss: {:.4f}".format(test_loss))

    return test_loss


loss_list = []
test_loss_list = []
num_epochs = 20

# model.load_state_dict(torch.load('vae.pth'))

for epoch in range(1, num_epochs + 1):
    loss = train(epoch)
    test_loss = test(epoch)
    print(loss)
    print(test_loss)

    print(
        "epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}".format(
            epoch + 1, num_epochs, loss, test_loss
        )
    )

    # logging
    loss_list.append(loss)
    test_loss_list.append(test_loss)

    np.save("loss_list_baseline.npy", np.array(loss_list))
    np.save("test_loss_list_baseline.npy", np.array(test_loss_list))

    print(time.time() - start)

# save the training model
np.save("loss_list_baseline.npy", np.array(loss_list))
np.save("test_loss_list_baseline.npy", np.array(test_loss_list))
torch.save(model.state_dict(), "baseline.pth")

