import optuna
from os.path import join
import os

from util import parse
from vae import train_vae
from vqvae import train_vqvae


args = parse()


def objective(trial):

    num_layers = 2  # trial.suggest_int("num_lstm_layers", 1, 2)
    args.num_layers = num_layers

    z_dim = trial.suggest_categorical("z_dim", [1, 2, 8])
    args.z_dim = z_dim

    if args.output_dir.find("/") >= 0:
        args.output_dir = args.output_dir[: args.output_dir.index("/")]

    if args.quantized:
        num_class = 2  # trial.suggest_int("num_class", 2 , 4)
        args.num_class = num_class
        output_dir_path = join(
            args.output_dir, "{}layers_zdim{}_nc{}".format(num_layers, z_dim, num_class)
        )
    else:
        output_dir_path = join(
            args.output_dir, "{}layers_zdim{}".format(num_layers, z_dim)
        )

    os.makedirs(output_dir_path, exist_ok=True)

    trian_func = train_vqvae if args.quantized else train_vae

    tmp_args = vars(args)
    tmp_args["output_dir"] = output_dir_path
    f0_loss = trian_func(tmp_args, trial=trial)

    return f0_loss


pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(pruner=pruner)
study.optimize(objective, n_trials=args.num_trials)

