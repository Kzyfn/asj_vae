import numpy as np
import argparse
from glob import glob
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from os.path import join
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
import torch
import torch.nn.functional as F

from models import BinaryFileSource
from loss_func import calc_lf0_rmse, rmse

mgc_dim = 180
lf0_dim = 3
vuv_dim = 1
bap_dim = 15
acoustic_linguisic_dim = 442
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim
mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse():
    parser = argparse.ArgumentParser(description="LSTM VAE",)
    parser.add_argument(
        "-ne", "--num_epoch", type=int, default=30,
    )
    parser.add_argument(
        "-nl", "--num_layers", type=int, default=1,
    )
    parser.add_argument(
        "-zd", "--z_dim", type=int, default=1,
    )
    parser.add_argument("-od", "--output_dir", type=str, required=True,),
    parser.add_argument("-dr", "--dropout_ratio", type=float, default=0.3,),
    parser.add_argument("-mp", "--model_path", type=str, default="",),
    parser.add_argument("-tr", "--train_ratio", type=float, default=1.0),
    parser.add_argument("-nc", "--num_class", type=int, default=1),
    parser.add_argument("-q", "--quantized", type=bool, default=False),
    parser.add_argument("-nt", "--num_trials", type=int, default=30)

    return parser.parse_args()


def train(epoch, model, train_loader, loss_function, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        tmp = []
        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).to(device))

        optimizer.zero_grad()
        recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2])
        loss = loss_function(
            recon_batch, tmp[1][:, lf0_start_idx], z_mu, z_unquantized_logvar
        )
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        del tmp

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader)
        )
    )

    return train_loss / len(train_loader)


def test(epoch, model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    f0_loss = 0
    with torch.no_grad():
        for i, data, in enumerate(test_loader):
            tmp = []

            for j in range(2):
                tmp.append(torch.tensor(data[j]).to(device))

            recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2])
            test_loss += loss_function(
                recon_batch, tmp[1][:, lf0_start_idx], z_mu, z_unquantized_logvar
            ).item()
            f0_loss += rmse(
                recon_batch.cpu().numpy().reshape(-1, 1),
                tmp[1].cpu().numpy().reshape(-1, 199)[:, lf0_start_idx],
            )
            del tmp

    test_loss /= len(test_loader)
    f0_loss /= len(test_loader)
    print("====> Test set loss: {:.4f}".format(test_loss))

    return test_loss, f0_loss


def create_loader(test=False):
    DATA_ROOT = "./data/basic5000"
    X = {"acoustic": {}}
    Y = {"acoustic": {}}
    utt_lengths = {"acoustic": {}}
    for ty in ["acoustic"]:
        for phase in ["train", "test"]:
            train = phase == "train"
            x_dim = (
                duration_linguistic_dim if ty == "duration" else acoustic_linguisic_dim
            )
            y_dim = duration_dim if ty == "duration" else acoustic_dim
            X[ty][phase] = FileSourceDataset(
                BinaryFileSource(
                    join(DATA_ROOT, "X_{}".format(ty)), dim=x_dim, train=train
                )
            )
            Y[ty][phase] = FileSourceDataset(
                BinaryFileSource(
                    join(DATA_ROOT, "Y_{}".format(ty)), dim=y_dim, train=train
                )
            )
            utt_lengths[ty][phase] = np.array(
                [len(x) for x in X[ty][phase]], dtype=np.int
            )

    X_min = {}
    X_max = {}
    Y_mean = {}
    Y_var = {}
    Y_scale = {}

    for typ in ["acoustic"]:
        X_min[typ], X_max[typ] = minmax(X[typ]["train"], utt_lengths[typ]["train"])
        Y_mean[typ], Y_var[typ] = meanvar(Y[typ]["train"], utt_lengths[typ]["train"])
        Y_scale[typ] = np.sqrt(Y_var[typ])

    mora_index_lists = sorted(glob(join("data/basic5000/mora_index", "squeezed_*.csv")))
    mora_index_lists_for_model = [
        np.loadtxt(path).reshape(-1) for path in mora_index_lists
    ]

    train_mora_index_lists = []
    test_mora_index_lists = []

    for i, mora_i in enumerate(mora_index_lists_for_model):
        if (i - 1) % 20 == 0:  # test
            pass
        elif i % 20 == 0:  # valid
            test_mora_index_lists.append(mora_i)
        else:
            train_mora_index_lists.append(mora_i)

    X_acoustic_train = [
        minmax_scale(
            X["acoustic"]["train"][i],
            X_min["acoustic"],
            X_max["acoustic"],
            feature_range=(0.01, 0.99),
        )
        for i in range(len(X["acoustic"]["train"]))
    ]
    Y_acoustic_train = [
        scale(Y["acoustic"]["train"][i], Y_mean["acoustic"], Y_scale["acoustic"])
        for i in range(len(Y["acoustic"]["train"]))
    ]
    train_mora_index_lists = [
        train_mora_index_lists[i] for i in range(len(train_mora_index_lists))
    ]

    X_acoustic_test = [
        minmax_scale(
            X["acoustic"]["test"][i],
            X_min["acoustic"],
            X_max["acoustic"],
            feature_range=(0.01, 0.99),
        )
        for i in range(len(X["acoustic"]["test"]))
    ]
    Y_acoustic_test = [
        scale(Y["acoustic"]["test"][i], Y_mean["acoustic"], Y_scale["acoustic"])
        for i in range(len(Y["acoustic"]["test"]))
    ]
    test_mora_index_lists = [
        test_mora_index_lists[i] for i in range(len(test_mora_index_lists))
    ]

    train_loader = [
        [X_acoustic_train[i], Y_acoustic_train[i], train_mora_index_lists[i]]
        for i in range(len(train_mora_index_lists))
    ]
    test_loader = [
        [X_acoustic_test[i], Y_acoustic_test[i], test_mora_index_lists[i]]
        for i in range(len(test_mora_index_lists))
    ]

    return train_loader, test_loader
