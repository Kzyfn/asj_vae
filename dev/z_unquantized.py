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
import IPython
from IPython.display import Audio

import matplotlib.pyplot as plt


from torch.utils import data as data_utils


import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tnrange, tqdm
from torch import optim
import torch.nn.functional as F


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


acoustic_linguisic_dim_model = 443


class Rnn(nn.Module):
    def __init__(self, bidirectional=True, num_layers=2):
        super(Rnn, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        ##ここまでエンコーダ

        self.fc11 = nn.Linear(
            acoustic_linguisic_dim_model, acoustic_linguisic_dim_model
        )

        self.lstm2 = nn.LSTM(
            acoustic_linguisic_dim_model,
            512,
            num_layers,
            bidirectional=bidirectional,
            dropout=0.15,
        )
        self.fc3 = nn.Linear(self.num_direction * 512, acoustic_dim)

    def decode(self, linguistic_features):
        x = self.fc11(linguistic_features.view(linguistic_features.size()[0], 1, -1))
        x = F.relu(x)
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)

        return self.fc3(h3)  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features):

        return self.decode(linguistic_features)


from models import VAE, VQVAE
from util import create_loader

train_loader, valid_loader = create_loader(valid=True)
train_loader, test_loader = create_loader(valid=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = VQVAE(num_layers=2, z_dim=1, num_class=4, repeat=1).to(device)
# model.load_state_dict(torch.load('vqvae_2_1_4_93dim/2layers_zdim1_nc4/vqvae_model_30.pth', map_location=torch.device('cpu')))
model.load_state_dict(
    torch.load("vqvae_2_1_4/vqvae_model_f0.pth", map_location=torch.device("cpu"))
)

print(model.quantized_vectors.weight)

h_l_labels_train = []
h_l_labels_test = []
h_l_labels_valid = []
for i in range(5000):
    h_l_label = np.loadtxt(
        "./data/basic5000/accents/accents_"
        + "0" * (4 - len(str(i + 1)))
        + str(i + 1)
        + ".csv",
    )
    if (i - 1) % 20 == 0:
        h_l_labels_test.append(h_l_label)
    elif i % 20 == 0:
        h_l_labels_valid.append(h_l_label)


def recon(index, model=model, z0=False, valid=False, lh=False, verbose=True):
    with torch.no_grad():
        tmp = []
        data = test_loader[index]
        if valid:
            data = valid_loader[index]
        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).float().to(device))
        h_l_label = h_l_labels_test[index] if not valid else h_l_labels_valid[index]

        if z0:
            y = model.decode(
                torch.tensor([[1] * 1] * data[2].shape[0]), tmp[0], data[2]
            )
            return y
        elif lh:
            y = model.decode(
                torch.from_numpy(h_l_label * 28 - 14).float(), tmp[0], data[2]
            )
            return y
        y, mu, logvar = model(tmp[0], tmp[1], data[2], 0)
        # print(mu.size())
        # print(h_l_labels_test[index].shape)

        if verbose:
            xlis = [
                "MO",
                "KU",
                "YO",
                "O(U)",
                "BI",
                "TE",
                "I",
                "SE",
                "N",
                "KA",
                "I",
                "DA",
                "N",
                "WA",
                "NA",
                "N",
                "NO",
                "SHI",
                "N",
                "TE",
                "N",
                "MO",
                "NA",
                "I",
                "MA",
                "MA",
                "SHU",
                "U",
                "RYO",
                "O(U)",
                "SHI",
                "MA",
                "SHI",
                "TA",
            ]
            plt.figure(figsize=(20, 6))

            plt.plot(mu.view(-1).cpu().numpy())
            plt.plot(h_l_label * 28 - 14)
            plt.xticks(list(range(len(h_l_label))), xlis)

            plt.show()
            print(np.corrcoef([mu.view(-1).cpu().numpy(), h_l_label])[0][1])

    return y, mu, logvar


def rmse(A, B):
    return np.sqrt((np.square(A - B)).mean())


def calc_lf0_rmse(natural, generated, lf0_idx=lf0_start_idx, vuv_idx=vuv_start_idx):
    idx = (natural[:, vuv_idx]).astype(bool)
    return (
        rmse(natural[idx, lf0_idx], generated[idx]) * 1200 / np.log(2)
    )  # unit: [cent]


ans = []
for i in range(250):
    y, z_q, z_uq = recon(i, verbose=False)
    ans.append([z_q, h_l_labels_test[i], z_uq])

import pickle

f = open("vqvae_z.txt", "wb")

pickle.dump(ans, f)
