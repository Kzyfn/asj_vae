import numpy as np
import argparse

def parse():
    parser = argparse.ArgumentParser(description='LSTM VAE', )
    parser.add_argument(
        '-ne',
        '--num_epoch',
        type=int,
        default=30,
    )
    parser.add_argument(
        '-nl',
        '--num_lstm_layers',
        type=int,
        #required=True,
        default=1,
    )
    parser.add_argument(
        '-zd',
        '--z_dim',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-od',
        '--output_dir',
        type=str,
        required=True,
    ),
    parser.add_argument(
        '-dr',
        '--dropout_ratio',
        type=float,
        default=0.3,
    ),
    parser.add_argument(
        '-mp',
        '--model_path',
        type=str,
        default='',
    ),
    parser.add_argument(
        '-tr',
        '--train_ratio',
        type=float,
        default=1.0
    ),
    parser.add_argument(
        '_nc',
        '--num_class',
        type=int,
        default=1
    )

    return parser.parse_args()


def create_loader(x_train, x_test, y_train, y_test, mora_i_train, mora_i_test):
    X_acoustic_train = [x_train['acoustic']['train'][i] for i in range(len(x_train['acoustic']['train']))]
    Y_acoustic_train = [y_train['acoustic']['train'][i] for i in range(len(y_train['acoustic']['train']))]
    train_mora_index_lists = [mora_i_train[i] for i in range(len(mora_i_train))]


    X_acoustic_test = [x_test['acoustic']['test'][i] for i in range(len(x_test['acoustic']['test']))]
    Y_acoustic_test = [y_test['acoustic']['test'][i] for i in range(len(y_test['acoustic']['test']))]
    test_mora_index_lists = [mora_i_test[i] for i in range(len(mora_i_test))]

    train_loader = [[X_acoustic_train[i], Y_acoustic_train[i], train_mora_index_lists[i]] for i in range(len(mora_i_train))]
    test_loader = [[X_acoustic_test[i], Y_acoustic_test[i], test_mora_index_lists[i]] for i in range(len(mora_i_test))]

    return train_loader, test_loader

def train(epoch, model, train_loader, loss_function, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        tmp = []
        for j in range(2):
            tmp.append(torch.from_numpy(data[j]).to(device))

        optimizer.zero_grad()
        recon_batch, z_mu, z_unquantized_logvar = model(tmp[0], tmp[1], data[2])
        loss = loss_function(recon_batch, tmp[1], z_mu, z_unquantized_logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        del tmp

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))
    
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
            test_loss += loss_function(recon_batch, tmp[1], z_mu, z_unquantized_logvar).item()
            f0_loss += calc_lf0_rmse(recon_batch.cpu().numpy().reshape(-1, 199), tmp[1].cpu().numpy().reshape(-1, 199), lf0_start_idx, vuv_start_idx)
            del tmp

    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss / len(test_loader)))
    
    return test_loss, f0_loss