import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tnrange, tqdm
import optuna
import os
import pickle

from models import Accent_Rnn, BinaryFileSource, VQVAE
from loss_func import calc_lf0_rmse, vae_loss
from util import create_loader, train, test, parse

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(epoch, model, train_loader, z_train, optimizer):
    model.train()
    train_loss = 0
    f0_loss = 0
    train_pred_z = []
    for batch_idx, data in enumerate(train_loader):
        tmp = []
        for j in range(1):
            tmp.append(torch.from_numpy(data[j]).float().to(device))
        with torch.autograd.detect_anomaly():
            optimizer.zero_grad()
            z_pred = model(tmp[0], data[2])
            loss = F.mse_loss(
                z_pred.view(-1),
                torch.from_numpy(z_train[batch_idx]).float().to(device),
            )

            loss.backward()
            train_loss += loss.item()
            print(loss.item())
            if torch.isnan(loss):
                print(z_pred)
                print(z_pred.size())
                print(z_train[batch_idx])
                print(z_train[batch_idx].shape)
            optimizer.step()

        del tmp
        train_pred_z.append(z_pred.detach().cpu().numpy().reshape(-1))

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader)
        )
    )
    return train_loss / len(train_loader), train_pred_z


def test(epoch, model, test_loader, z_test):
    model.eval()
    test_loss = 0
    f0_loss = 0
    test_pred_z = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            tmp = []
            for j in range(1):
                tmp.append(torch.from_numpy(data[j]).float().to(device))

            z_pred = model(tmp[0], data[2])
            loss = F.mse_loss(
                z_pred.view(-1), torch.from_numpy(z_test[batch_idx]).float().to(device),
            )
            test_loss += loss.item()
            del tmp

            test_pred_z.append(z_pred.detach().cpu().numpy().reshape(-1))

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, test_loss / len(test_loader)
        )
    )
    return test_loss / len(test_loader), test_pred_z


def train_accent_rnn(args, trial=None, test_ratio=1):
    """

    """
    model = Accent_Rnn().to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)  # 1e-3
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    train_loader, test_loader = create_loader()
    train_num = int(args["train_ratio"] * len(train_loader))  # 1
    test_num = int(test_ratio * len(test_loader))
    train_loader = train_loader[:train_num]
    test_loader = test_loader[:test_num]

    file_not_exists = not os.path.isfile("z_train.csv")

    if file_not_exists:
        vqvae_model = VQVAE(
            num_layers=2,  # args["num_layers"],
            z_dim=1,  # args["z_dim"],
            num_class=4,  # args["num_class"],
        ).to(device)
        vqvae_model.load_state_dict(torch.load("vqvae_model.pth"))
        z_train = []
        z_test = []
        with torch.no_grad():
            for idx, data in enumerate(tqdm(train_loader)):
                tmp = []
                for j in range(2):
                    tmp.append(torch.from_numpy(data[j]).float().to(device))
                isnan = True

                while isnan:
                    recon_batch, z, z_unquantized = vqvae_model(
                        tmp[0], tmp[1], data[2], 0
                    )

                    isnan = np.isnan(z.detach().cpu().numpy()).any()
                    if isnan:
                        print(idx)
                        print(z)

                z_train.append(z.cpu().numpy().reshape(-1))

            for data in tqdm(test_loader):
                tmp = []
                for j in range(2):
                    tmp.append(torch.from_numpy(data[j]).float().to(device))
                isnan = True

                while isnan:
                    recon_batch, z, z_unquantized = vqvae_model(
                        tmp[0], tmp[1], data[2], 0
                    )
                    isnan = np.isnan(z.detach().cpu().numpy()).any()
                z_test.append(z.cpu().numpy().reshape(-1))

        with open("z_train.csv", mode="wb") as f:
            pickle.dump(z_train, f)

        with open("z_test.csv", mode="wb") as f:
            pickle.dump(z_test, f)

    else:
        with open("z_train.csv", mode="rb") as f:
            z_train = pickle.load(f)
        with open("z_test.csv", mode="rb") as f:
            z_test = pickle.load(f)

    loss_list = []
    test_loss_list = []
    f0_loss_list = []
    f0_loss_trainlist = []
    start = time.time()

    for epoch in range(1, args["num_epoch"] + 1):
        loss, train_pred_z = train(epoch, model, train_loader, z_train, optimizer)
        test_loss, test_pred_z = test(epoch, model, test_loader, z_test)
        # scheduler.step()
        print(
            "epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}".format(
                epoch + 1, args["num_epoch"], loss, test_loss
            )
        )

        # logging
        loss_list.append(loss)
        test_loss_list.append(test_loss)

        if trial is not None:
            trial.report(test_loss, epoch - 1)

        if trial is not None:
            if trial.should_prune():
                return optuna.TrialPruned()

        print(time.time() - start)

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                args["output_dir"] + "/vae_model_{}.pth".format(epoch),
            )
            with open(
                args["output_dir"] + "train_pred_z_{}.csv".format(epoch), mode="wb"
            ) as f:
                pickle.dump(train_pred_z, f)

            with open(
                args["output_dir"] + "test_pred_z{}.csv".format(epoch), mode="wb"
            ) as f:
                pickle.dump(test_pred_z, f)

        np.save(args["output_dir"] + "/loss_list.npy", np.array(loss_list))
        np.save(args["output_dir"] + "/test_loss_list.npy", np.array(test_loss_list))

    return f0_loss


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    train_accent_rnn(vars(args))
