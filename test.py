import os
import torch
import numpy as np
import sklearn
import tqdm

import utils
import model
import loss_function

# load yaml
param = utils.load_yaml()


def evalute_test_machine(model_name, target_dir, pool_type='mean'):
    global csv_lines
    # hyperparameter
    n_mels = param['n_mels']
    frames = param['frames']
    result_dir = param['result_dir']

    machine_type = os.path.split(target_dir)[1]
    # result csv
    csv_lines.append([machine_type])
    csv_lines.append(['id', 'AUC', 'pAUC'])
    performance = []
    # load model
    model_dir = param['model_dir']
    model_path = f'{model_dir}/{model_name}/{machine_type}/model_{machine_type}.pkl'
    if model_name == 'AE':
        ae_net = model.Auto_encoder(input_dim=frames*n_mels, output_dim=frames*n_mels)
    elif model_name == 'IDNN':
        ae_net = model.Auto_encoder_small(input_dim=(frames - 1) * n_mels, output_dim=n_mels)
    else:
        raise Exception(f'The model name {model_name} mismatch with "AE and IDNN"')
    ae_net.load_state_dict(torch.load(model_path).module.state_dict())

    # get machine list
    machine_id_list = utils.get_machine_id_list(target_dir)

    for id_str in machine_id_list:
        test_files, y_true = utils.create_test_file_list(target_dir, id_str)
        anomaly_score_csv = f'{result_dir}/anomaly_score_{machine_type}_{id_str}.csv'
        anomaly_score_list = []
        y_pred = [0. for _ in test_files]
        for file_idx, file_path in tqdm.tqdm(enumerate(test_files), total=len(test_files)):
            data = utils.log_mel_spect_to_vector(file_path,
                                                 n_mels=param['n_mels'],
                                                 frames=param['frames'],
                                                 n_fft=param['n_fft'],
                                                 hop_length=param['hop_length'],
                                                 power=param['power'])
            data = torch.from_numpy(data).float()

            if model_name == 'AE':
                model_input = data
                model_target = data
            elif model_name == 'IDNN':
                model_input = torch.zeros((data.size(0), n_mels * (frames - 1)))
                model_input[:, :n_mels * (frames // 2)] = data[:, :n_mels * (frames // 2)]
                model_input[:, n_mels * (frames // 2):] = data[:, n_mels * (frames // 2 + 1):]
                model_target = data[:, n_mels * (frames // 2): n_mels * (frames // 2 + 1)]
            else:
                raise Exception(f'The model name {model_name} mismatch with "AE and IDNN"')

            with torch.no_grad():
                ae_net.eval()
                predict_data, _ = ae_net(model_input)

            model_target = model_target.numpy()
            predict_data = predict_data.numpy()

            if model_name == 'AE':
                y_pred[file_idx] = utils.calculate_anomaly_score(param, model_target, predict_data,
                                                                 frames=frames, n_mels=n_mels,
                                                                 pool_type=pool_type)
            elif model_name == 'IDNN':
                y_pred[file_idx] = utils.calculate_anomaly_score(param, model_target, predict_data,
                                                                 frames=1, n_mels=n_mels,
                                                                 pool_type=pool_type)
            else:
                raise Exception(f'The model name {model_name} mismatch with "AE and IDNN"')
            anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]] )
        utils.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        # compute Auc and pAuc
        auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
        max_fpr = param['max_fpr']
        p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
        csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
        performance.append([auc, p_auc])
    # calculate averages for AUCs and pAUCs
    averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(['Average'] + list(averaged_performance))
    csv_lines.append([])


if __name__ == '__main__':
    os.makedirs(param['result_dir'], exist_ok=True)
    result_dir = param['result_dir']
    avg_type = ['mean', 'gwrp', 'max']
    process_mt_list = ['pump', 'a_fan', 'slider', 'valve']
    dirs = utils.select_dirs(param)
    csv_lines = []
    for pt in avg_type:
        for idx, target_dir in enumerate(dirs):
            machine_type = os.path.split(target_dir)[-1]
            if machine_type not in process_mt_list:
                continue
            print('\n' + '='*20)
            print(f'[{idx+1}/{len(dirs)}] {target_dir} [{pt}]')
            evalute_test_machine('AE', target_dir, pool_type=pt)
        result_file_name = param['result_file']
        result_path = f'{result_dir}/{result_file_name}'
    utils.save_csv(save_file_path=result_path, save_data=csv_lines)
