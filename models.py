import torch
from torch import nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import numpy as np
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
from sklearn.cluster import KMeans
import random

mgc_dim = 180  # メルケプストラム次数　？？
lf0_dim = 3  # 対数fo　？？ なんで次元が３？
vuv_dim = 1  # 無声or 有声フラグ　？？
bap_dim = 15  # 発話ごと非周期成分　？？

duration_linguistic_dim = 438  # question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442  # 上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim  # aoustice modelで求めたいもの

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]


device = "cuda" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    def __init__(self, num_layers, z_dim, bidirectional=True, dropout=0.3):
        super(VAE, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.z_dim = z_dim
        self.fc11 = nn.Linear(
            acoustic_linguisic_dim + acoustic_dim, acoustic_linguisic_dim + acoustic_dim
        )

        self.lstm1 = nn.LSTM(
            acoustic_linguisic_dim + acoustic_dim,
            400,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )  # 入力サイズはここできまる
        self.fc21 = nn.Linear(self.num_direction * 400, z_dim)
        self.fc22 = nn.Linear(self.num_direction * 400, z_dim)
        ##ここまでエンコーダ

        self.fc12 = nn.Linear(
            acoustic_linguisic_dim + z_dim, acoustic_linguisic_dim + z_dim
        )
        self.lstm2 = nn.LSTM(
            acoustic_linguisic_dim + z_dim,
            400,
            2,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.fc3 = nn.Linear(self.num_direction * 400, acoustic_dim)

    def encode(self, linguistic_f, acoustic_f, mora_index, batch_size=1):
        x = torch.cat([linguistic_f, acoustic_f], dim=1)
        x = self.fc11(x)
        x = F.relu(x)

        out, hc = self.lstm1(x.view(x.size()[0], 1, -1))
        out_forward = out[:, :, :400][mora_index]
        mora_index_for_back = np.concatenate([[0], mora_index[:-1] + 1])
        out_back = out[:, :, 400:][mora_index_for_back]
        out = torch.cat([out_forward, out_back], dim=2)

        h1 = F.relu(out)

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, linguistic_features, mora_index):

        z_tmp = torch.tensor(
            [[0] * self.z_dim] * linguistic_features.size()[0],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device)

        for i, mora_i in enumerate(mora_index):
            prev_index = 0 if i == 0 else int(mora_index[i - 1])
            z_tmp[prev_index : int(mora_i)] = z[i]

        x = torch.cat([linguistic_features, z_tmp.view(-1, self.z_dim)], dim=1)
        x = self.fc12(x)
        x = F.relu(x)

        h3, (h, c) = self.lstm2(x.view(linguistic_features.size()[0], 1, -1))
        h3 = F.relu(h3)

        return self.fc3(h3)  # torch.sigmoid(self.fc3(h3))

    def forward(
        self, linguistic_features, acoustic_features, mora_index, epoch
    ):  # epochはVQVAEと合わせるため
        mu, logvar = self.encode(linguistic_features, acoustic_features, mora_index)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, linguistic_features, mora_index), mu, logvar


class VQVAE(nn.Module):
    def __init__(
        self, bidirectional=True, num_layers=2, num_class=2, z_dim=1, dropout=0.3
    ):
        super(VQVAE, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.num_class = num_class
        self.quantized_vectors = nn.Embedding(
            num_class, z_dim
        )  # torch.tensor([[i]*z_dim for i in range(nc)], requires_grad=True)
        # self.quantized_vectors.weight.data.uniform_(0, 1)
        self.quantized_vectors.weight = nn.init.normal_(
            self.quantized_vectors.weight, 0.1, 0.001
        )

        self.z_dim = z_dim

        self.fc11 = nn.Linear(
            acoustic_linguisic_dim + acoustic_dim, acoustic_linguisic_dim + acoustic_dim
        )
        self.lstm1 = nn.LSTM(
            acoustic_linguisic_dim + acoustic_dim,
            400,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )  # 入力サイズはここできまる
        self.fc2 = nn.Linear(self.num_direction * 400, z_dim)
        ##ここまでエンコーダ

        self.fc12 = nn.Linear(
            acoustic_linguisic_dim + z_dim, acoustic_linguisic_dim + z_dim
        )
        self.lstm2 = nn.LSTM(
            acoustic_linguisic_dim + z_dim,
            400,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc3 = nn.Linear(self.num_direction * 400, acoustic_dim)

    def choose_quantized_vector(self, z, epoch):  # zはエンコーダの出力
        if epoch >= 1:
            with torch.no_grad():
                error = torch.sum((self.quantized_vectors.weight - z) ** 2, dim=1)
                min_index = torch.argmin(error).item()
        else:
            min_index = random.choice(list(range(self.num_class)))

        return self.quantized_vectors.weight[min_index]

    def quantize_z(self, z_unquantized, epoch):
        z = torch.zeros(z_unquantized.size(), requires_grad=True).to(device)

        for i in range(z_unquantized.size()[0]):
            z[i] = self.choose_quantized_vector(z_unquantized[i].reshape(-1), epoch)

        return z

    def encode(self, linguistic_f, acoustic_f, mora_index):
        x = torch.cat([linguistic_f, acoustic_f], dim=1)
        x = self.fc11(x)
        x = F.relu(x)
        out, hc = self.lstm1(x.view(x.size()[0], 1, -1))
        out_forward = out[:, :, :400][mora_index]
        mora_index_for_back = np.concatenate([[0], mora_index[:-1] + 1])
        out_back = out[:, :, 400:][mora_index_for_back]
        out = torch.cat([out_forward, out_back], dim=2)

        h1 = F.relu(out)

        return self.fc2(h1)

    def init_codebook(self, codebook):
        self.quantized_vectors.weight = codebook

    def decode(self, z, linguistic_features, mora_index):

        z_tmp = torch.tensor(
            [[0] * self.z_dim] * linguistic_features.size()[0],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device)

        for i, mora_i in enumerate(mora_index):
            prev_index = 0 if i == 0 else int(mora_index[i - 1])
            z_tmp[prev_index : int(mora_i)] = z[i]

        x = torch.cat([linguistic_features, z_tmp.view(-1, self.z_dim)], dim=1).view(
            linguistic_features.size()[0], 1, -1
        )

        x = self.fc12(x)
        x = F.relu(x)

        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)

        return self.fc3(h3)  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features, acoustic_features, mora_index, epoch):
        z_not_quantized = self.encode(
            linguistic_features, acoustic_features, mora_index
        )
        # print(z_not_quantized)
        z = self.quantize_z(z_not_quantized, epoch)

        return self.decode(z, linguistic_features, mora_index), z, z_not_quantized


class Rnn(nn.Module):
    def __init__(
        self, bidirectional=True, num_layers=1, accent_label_type=0
    ):  # accent_label_type: 0;なし, 1; 92次元, 2; H/Lラベル
        super(Rnn, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        ##ここまでエンコーダ

        if accent_type == 0:
            acoustic_linguisic_dim_ = acoustic_linguisic_dim
        elif accent_type == 2:
            acoustic_linguisic_dim_ = acoustic_linguisic_dim + 1
        else:
            acoustic_linguisic_dim_ = 535

        self.lstm2 = nn.LSTM(
            acoustic_linguisic_dim_, 400, num_layers, bidirectional=bidirectional
        )
        self.fc3 = nn.Linear(self.num_direction * 400, acoustic_dim)

    def decode(self, linguistic_features):
        x = linguistic_features.view(linguistic_features.size()[0], 1, -1)
        h3, (h, c) = self.lstm2(x)
        h3 = F.relu(h3)

        return self.fc3(h3)  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features):

        return self.decode(linguistic_features)


class BinaryFileSource(FileDataSource):
    def __init__(self, data_root, dim, train):
        self.data_root = data_root
        self.dim = dim
        self.train = train

    def collect_files(self):
        files = sorted(glob(join(self.data_root, "*.bin")))
        # files = files[:len(files)-5] # last 5 is real testset
        train_files = []
        test_files = []
        # train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

        for i, path in enumerate(files):
            if (i - 1) % 20 == 0:  # test
                pass
            elif i % 20 == 0:  # valid
                test_files.append(path)
            else:
                train_files.append(path)

        if self.train:
            return train_files
        else:
            return test_files

    def collect_features(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, self.dim)


class LBG:
    def __init__(self, num_class=2, z_dim=8):
        self.num_class = num_class
        self.z_dim = z_dim
        self.eps = np.array([1e-3] * z_dim)

    def calc_center(self, x):
        vectors = x.view(-1, self.z_dim)
        center_vec = torch.sum(vectors, dim=0) / vectors.size()[0]

        return center_vec

    def calc_q_vec_init(self, x):
        center_vec = self.calc_center(x).cpu().numpy()
        init_rep_vecs = np.array([center_vec - self.eps, center_vec + self.eps])

        return init_rep_vecs

    def calc_q_vec(self, x):
        # はじめに最初の代表点を求める
        init_rep_vecs = self.calc_q_vec_init(x)
        # K-means で２クラスに分類
        data = x.cpu().numpy()
        kmeans = KMeans(n_clusters=2, init=init_rep_vecs).fit(data)

        rep_vecs = kmeans.cluster_centers_

        for i in range(int(np.log2(self.num_class)) - 1):
            kmeans = KMeans(n_clusters=2 ** (i + 1), init=rep_vecs)
            rep_vecs = kmeans.cluster_centers_

        return rep_vecs

