import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

import utils
import dataset
import model
import loss_function

# load yaml
param = utils.load_yaml()

# train AE
def train(train_loader,
             machine_type,
             epochs=param['epochs'],
             every_epochs_save=param['every_epochs_save'],
             lr=param['lr'],
             cuda=param['cuda'],
             device_ids=param['device_ids'],
             model_name='AE'):
    # hyperparameter
    n_mels = param['n_mels']
    frames = param['frames']
    # set file path
    model_dir = param['model_dir']
    model_save_dir = f'{model_dir}/{model_name}'
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(f'{model_save_dir}/{machine_type}', exist_ok=True)

    # get model
    if model_name == 'AE':
        ae_net = model.Auto_encoder(input_dim=frames*n_mels, output_dim=frames*n_mels)
    elif model_name == 'IDNN':
        ae_net = model.Auto_encoder_small(input_dim=(frames - 1) * n_mels, output_dim=n_mels)
    else:
        raise Exception(f'The model name {model_name} mismatch with "AE and IDNN"')
    if cuda:
        ae_net = nn.DataParallel(ae_net, device_ids=device_ids)
        ae_net.cuda()

    # set loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_net.parameters(), lr=lr)

    # train
    loss_list = []
    sum = len(train_loader)
    for epoch in range(1, epochs + 1):
        ls = 0
        mse = 0
        for iter, (inputs, target) in enumerate(train_loader):
            inputs = inputs.float()
            target = target.float()

            if model_name == 'AE':
                model_input = inputs
                model_target = target
            elif model_name == 'IDNN':
                model_input = torch.zeros((inputs.size(0), n_mels * (frames - 1)))
                model_input[:, :n_mels * (frames // 2)] = inputs[:, :n_mels * (frames // 2)]
                model_input[:, n_mels * (frames // 2):] = inputs[:, n_mels * (frames // 2 + 1):]
                model_target = target[:, n_mels * (frames // 2): n_mels * (frames // 2 + 1)]
            else:
                raise Exception(f'The model name {model_name} mismatch with "AE and IDNN"')
            if cuda:
                model_input = model_input.cuda()
                model_target = model_target.cuda()
            #outputs, features = ae_net(inputs)
            outputs, _ = ae_net(model_input)
            #loss = criterion(outputs, target, mu, logvar)
            loss = criterion(outputs, model_target)
            ls += loss.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ls /= sum
        mse /= sum
        loss_list.append(ls)
        print(f'[{epoch}/{epochs}] Loss: {ls:.5f} | mse: {mse: .5f}')
        if epoch % every_epochs_save == 0:
            model_file_path = f'{model_save_dir}/{machine_type}/model_{machine_type}_{epoch}.pkl'
            torch.save(ae_net, model_file_path)

if __name__ == '__main__':
    os.makedirs(param['model_dir'], exist_ok=True)
    dirs = utils.select_dirs(param)
    process_mt_list = ['pump', 'a_fan', 'slider', 'valve']
    for target_dir in dirs:
        machine_type = os.path.split(target_dir)[-1]
        if machine_type not in process_mt_list:
            continue
        pre_data_path = param['pre_data_dir'] + f'/{machine_type}.db'
        print(f'loading dataset [{machine_type}] ......')
        my_dataset = dataset.MyDataset(pre_data_path, keys=['log_mel'])
        train_loader = DataLoader(my_dataset, batch_size=param['batch_size'], shuffle=True)
        print('training ......')
        train(train_loader, machine_type, model_name='AE')


