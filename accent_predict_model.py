import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tnrange, tqdm
import optuna
import os

from models import Accent_Rnn, BinaryFileSource, VQVAE
from loss_func import calc_lf0_rmse, vae_loss
from util import create_loader, train, test, parse

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(epoch, model, train_loader, z_train, optimizer):
    model.train()
    train_loss = 0
    f0_loss = 0
    for batch_idx, data in enumerate(train_loader):
        tmp = []
        for j in range(1):
            tmp.append(torch.from_numpy(data[j]).float().to(device))

        optimizer.zero_grad()
        z_pred = model(tmp[0], data[2])
        loss = F.mse_loss(
            z_pred.view(-1), torch.from_numpy(z_train[batch_idx]).to(device)
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


def test(epoch, model, test_loader, z_test):
    model.eval()
    train_loss = 0
    f0_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            tmp = []
            for j in range(1):
                tmp.append(torch.from_numpy(data[j]).float().to(device))

            z_pred = model(tmp[0], data[2])
            loss = F.mse_loss(
                z_pred.view(-1), torch.from_numpy(z_train[batch_idx]).to(device)
            )
            test_loss += loss.item()
            del tmp

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, test_loss / len(test_loader)
        )
    )
    return test_loss / len(test_loader)


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
        vqvae_model = VQVAE(num_layers=args["num_layers"], z_dim=args["z_dim"], num_class=args["num_class"]num_layers=args["num_layers"], z_dim=args["z_dim"], num_class=args["num_class"]).to(device)
        vqvae_model.load_state_dict(torch.load("vqvae_model.pth"))
        z_train = []
        z_test = []
        with torch.no_grad():
            tmp = []
            for data in tqdm(train_loader):
                for x in range(2):
                    tmp.append(torch.from_numpy(data[j]).float().to(device))
                    recon_batch, z, z_unquantized = model(tmp[0], tmp[1], data[2], 0)
                    z_train.append(z)

            for data in tqdm(test_loader):
                for x in range(2):
                    tmp.append(torch.from_numpy(data[j]).float().to(device))
                    recon_batch, z, z_unquantized = model(tmp[0], tmp[1], data[2], 0)
                    z_test.append(z)
        np.savetxt("z_train.csv", z_train)
        np.savetxt("z_test.csv", z_test)

    else:
        z_train = np.loadtxt("z_train.csv")
        z_test = np.loadtxt("z_test.csv")

    loss_list = []
    test_loss_list = []
    f0_loss_list = []
    f0_loss_trainlist = []
    start = time.time()

    for epoch in range(1, args["num_epoch"] + 1):
        loss = train(epoch, model, train_loader, z_train, optimizer)
        test_loss, f0_loss = test(epoch, model, test_loader, z_test)
        # scheduler.step()
        print(
            "epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}".format(
                epoch + 1, args["num_epoch"], loss, test_loss
            )
        )

        # logging
        loss_list.append(loss)
        f0_loss_trainlist.append(f0_loss_train)
        test_loss_list.append(test_loss)
        f0_loss_list.append(f0_loss)

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
        np.save(args["output_dir"] + "/loss_list.npy", np.array(loss_list))
        np.save(args["output_dir"] + "/f0loss_list.npy", np.array(f0_loss_trainlist))
        np.save(args["output_dir"] + "/test_loss_list.npy", np.array(test_loss_list))
        np.save(args["output_dir"] + "/test_f0loss_list.npy", np.array(f0_loss_list))

    return f0_loss


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    train_accent_rnn(vars(args))
