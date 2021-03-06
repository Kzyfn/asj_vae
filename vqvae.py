import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tnrange, tqdm
import optuna
import os

from models import VQVAE, BinaryFileSource
from loss_func import calc_lf0_rmse, vqvae_loss
from util import create_loader, train, test, parse

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_vqvae(args, trial=None):
    """
    """
    model = VQVAE(
        num_layers=args["num_layers"], z_dim=args["z_dim"], num_class=args["num_class"]
    ).to(device)
    if args["model_path"] != "":
        model.load_state_dict(torch.load(args["model_path"]))

    optimizer = optim.Adam(model.parameters(), lr=2e-3)  # 1e-3

    train_loader, test_loader = create_loader()
    train_num = int(args["train_ratio"] * len(train_loader))  # 1
    train_loader = train_loader[:train_num]

    loss_list = []
    test_loss_list = []
    f0_loss_list = []
    start = time.time()
    for epoch in range(1, args["num_epoch"] + 1):
        loss = train(epoch, model, train_loader, vqvae_loss, optimizer)
        test_loss, f0_loss = test(epoch, model, test_loader, vqvae_loss)

        print(
            "epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}".format(
                epoch + 1, args["num_epoch"], loss, test_loss
            )
        )

        # logging
        loss_list.append(loss)
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
                args["output_dir"] + "/vqvae_model_{}.pth".format(epoch),
            )
        np.save(args["output_dir"] + "/loss_list.npy", np.array(loss_list))
        np.save(args["output_dir"] + "/test_loss_list.npy", np.array(test_loss_list))
        np.save(args["output_dir"] + "/test_f0loss_list.npy", np.array(f0_loss_list))

    return f0_loss


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    train_vqvae(vars(args), test_ratio=0.1)

