import numpy
import joblib
import os
import numpy as np
import glob
import tqdm

import utils

# preprocess and save pre_data file
def save_pre_data_file(param,
                       dir_name='train',
                       ext='wav',
                       n_mels=64,
                       frames=5,
                       n_fft=1024,
                       hop_length=512,
                       power=2.0):
    # data dirs
    dirs = utils.select_dirs(param)
    for index, target_dir in enumerate(sorted(dirs)):
        print('\n' + '='*20)
        print(f'[{index+1}/{len(dirs)}] {target_dir} preprocessing...')
        machine_type = os.path.split(target_dir)[1]
        pre_data_dir = param['pre_data_dir']
        os.makedirs(pre_data_dir, exist_ok=True)
        pre_data_file_list = glob.glob(f'{pre_data_dir}/*')
        save_file_path = f'{pre_data_dir}\\{machine_type}.db'
        if save_file_path in pre_data_file_list:
            print(f'{machine_type}[{index + 1}/{len(dirs)}] had saved')
            continue

        files = utils.create_file_list(target_dir,
                                       dir_name=dir_name,
                                       ext=ext)
        pre_data = {
            'log_mel': [],
        }
        for idx, file in tqdm.tqdm(enumerate(files), total=len(files)):
            vector = utils.log_mel_spect_to_vector(file,
                                                   n_mels=n_mels,
                                                   frames=frames,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   power=power)
            for i in range(vector.shape[0]):
                pre_data['log_mel'].append(vector[i])
            # center frame predict
            # for i in range(vector.shape[0]):
            #     log_mel = np.zeros((n_mels * (frames - 1)))
            #     log_mel[: n_mels * (frames // 2)] = vector[i][: n_mels * (frames // 2)]
            #     log_mel[n_mels * (frames // 2):] = vector[i][n_mels * (frames // 2 + 1):]
            #     #log_mel[n_mels*(frames // 2): n_mels * (frames // 2 + 1)] = np.zeros((n_mels,), dtype=np.float)
            #     pre_data['log_mel'].append(log_mel)
            #     pre_data['log_mel_next'].append(vector[i][n_mels*(frames // 2): n_mels * (frames // 2 + 1)])
            #     pre_data['id'].append(id_vector[i])
            # print(f'[{index+1}/{len(dirs)}][{idx+1}/{len(files)}] complete.')
        with open(save_file_path, 'wb') as f:
            joblib.dump(pre_data, f)
        print(f'{machine_type}[{index+1}/{len(dirs)}] had saved')


if __name__ == '__main__':
    save_pre_data_file(n_mels=param['n_mels'],
                       frames=param['frames'],
                       n_fft=param['n_fft'],
                       hop_length=param['hop_length'],
                       power=param['power'])
