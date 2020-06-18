import optuna


from util import parse
from vae import train_vae
from vqvae import train_vqvae

args = parse()


def objective(trial):

    num_layers = trial.suggest_int("num_lstm_layers", 1, 3)
    args.num_layers = num_layers

    z_dim = trial.suggest_categorical("z_dim", [1, 2, 16])
    args.z_dim = z_dim

    if args.quantized:
        num_class = trial.suggest_int("num_class", 2, 4)
        args.num_class = num_class

    trian_func = train_vqvae if args.quantized else train_vae

    f0_loss = trian_func(vars(args), trial=trial)

    return f0_loss


pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(pruner=pruner)
study.optimize(objective, n_trials=args.num_trials)

