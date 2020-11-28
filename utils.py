import os
import yaml
import sys
import logging
import glob
import csv
import re
import itertools

import numpy as np
import librosa

# recording log
logging.basicConfig(level=logging.DEBUG, filename='baseline.log')
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# load yaml (config file)
def load_yaml(file_name='./config.yaml'):
    with open(file_name) as f:
        param = yaml.safe_load(f)
    return param


# make log mel spectrogram for each file
def file_to_log_mel_spectrogram(file_name,
                                n_mels=64,
                                n_fft=1024,
                                hop_length=512,
                                power=2.0, ):
    y, sr = librosa.load(file_name, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram


# log mel spectrogram to vector
def log_mel_spect_to_vector(file_name,
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    """
    log mel spectrogram to vector
    :param frames: frames number concat as input
    :return: [vector_size, frames * n_mels]
    """
    log_mel_spect = file_to_log_mel_spectrogram(file_name,
                                                n_mels=n_mels,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
    dims = n_mels * frames
    vector_size = log_mel_spect.shape[1] - frames + 1
    if vector_size < 1:
        print('Warning: frames is too large!')
        return np.empty((0, dims))
    vector = np.zeros((vector_size, dims))
    for t in range(frames):
        vector[:, t * n_mels: (t + 1) * n_mels] = log_mel_spect[:, t: t + vector_size].T
    return vector

def log_mel_spect_to_vector_2d(file_name,
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0):
    """
    log mel spectrogram to 2d vector as input for CNN AE
    :return: [vector_size, 1, frames, n_mels]
    """
    log_mel_spect = file_to_log_mel_spectrogram(file_name,
                                                n_mels=n_mels,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
    vector_size = log_mel_spect.shape[1] - frames + 1
    if vector_size < 1:
        print('Warning: frames is too large!')
        return np.empty((0, 1, n_mels, frames))
    vector = np.zeros((vector_size, 1, n_mels, frames))
    for t in range(vector_size):
        vector[t, 0, :, :] = log_mel_spect[:, t: t+frames]
    return vector


# getting data dirs list
def select_dirs(param):
    logger.info('load_directory <- development')
    base_path = param['data_dir']
    dir_path = os.path.abspath(f'{base_path}/*')
    dirs = glob.glob(dir_path)
    return dirs


# getting file list from target dir
def create_file_list(target_dir,
                     dir_name='train',
                     ext='wav'):
    logger.info(f'target_dir: {target_dir}')
    list_path = os.path.abspath(f'{target_dir}/{dir_name}/*.{ext}')
    files = sorted(glob.glob(list_path))
    if len(files) == 0:
        logger.exception('no wav file !')
    logger.info(f'{dir_name}_file num: {len(files)}')
    return files


# save csv data
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


# getting machine id list from target dir
def get_machine_id_list(target_dir,
                        dir_name='test',
                        ext='wav'):
    dir_path = os.path.abspath(f'{target_dir}/{dir_name}/*.{ext}')
    files_path = sorted(glob.glob(dir_path))
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in files_path])
    )))
    return machine_id_list

# getting target dir file list and label list
def create_test_file_list(target_dir,
                          id_name,
                          dir_name='test',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


# def calculate_gwrp(errors, decay):
#     errors = sorted(errors, reverse=True)
#     gwrp_w = decay ** np.arange(len(errors))
#     #gwrp_w[gwrp_w < 0.1] = 0.1
#     sum_gwrp_w = np.sum(gwrp_w)
#     errors = errors * gwrp_w
#     errors = np.sum(errors)
#     score = errors / sum_gwrp_w
#     return score


# calculate anomaly score
def calculate_anomaly_score(param,
                            data,
                            predict_data,
                            frames=5,
                            n_mels=128,
                            pool_type='mean'):
    bs = data.shape[0]
    data = data.reshape(bs, frames, n_mels)
    predict_data = predict_data.reshape(bs, frames, n_mels)
    # mean score of n_mels for every frame
    errors = np.mean(np.square(data - predict_data), axis=2).reshape(bs, frames)
    # mean score of frames
    errors = np.mean(errors, axis=1)
    #errors = np.max(errors, axis=1)

    if pool_type == 'mean':
        score = np.mean(errors)
    elif pool_type == 'max':
        score = np.max(errors)
    elif pool_type == 'max_frames_mean':
        frames = param['frames']
        frames = errors.shape[0] // 5
        errors = sorted(errors, reverse=True)
        score = np.mean(errors[:frames])
    # elif pool_type == 'gwrp':
    #     decay = param['gwrp_decay']
    #     # len1 = len(errors)
    #     # len2 = len(errors[errors >= np.mean(errors)])
    #     # decay = (np.sum(errors[errors < np.mean(errors)]) + len2 * np.mean(errors)) / np.sum(errors)
    #     #print(decay)
    #     score = calculate_gwrp(errors, decay)
    elif pool_type == 'gtmean':
        errors = errors[errors > np.mean(errors)]
        # decay = param['gwrp_decay']
        # score = calculate_gwrp(errors, decay)
        score = np.mean(errors)
    else:
        raise Exception(f'the pooling type is {pool_type}, mismatch with mean, max, max_frames_mean, gwrp, and gt_mean')

    return score


if __name__ == '__main__':
    param = load_yaml(file_name='./config.yaml')
    file_name = os.path.join('./data', '*')
    files = glob.glob(file_name)
    print(files)
    # file_name = os.path.join(select_dirs(param)[0], 'train/normal_id_04_00000002.wav')
    # l_s, idv = log_mel_spect_to_vector(file_name, n_mels=128)
    # print(l_s.shape, idv.shape)
    # a = np.random.rand(64, 1, 128, 5)
    # b = np.random.rand(64, 1, 128, 5)
    # errors = np.mean(np.square(a - b), axis=1)
    # print(calculate_anomaly_score(param, a, b))
    # print(file_name)
    # vector = log_mel_spect_to_vector(file_name=file_name,
    #                                  n_mels=param['n_mels'],
    #                                  frames=param['frames'],
    #                                  n_fft=param['n_fft'],
    #                                  hop_length=param['hop_length'],
    #                                  power=param['power'])
    # print(np.max(vector), np.min(vector))

    # machine_li_list = get_machine_id_list('../dataset/pump')
    # print(machine_li_list)
    # files, labels = create_test_file_list('../dataset/pump', 'id_00')
    # print(files.shape, labels.shape)
    # print(files)
    # print(labels)

    # errors = np.random.random(10)
    # score = calculate_gwrp(errors, 0.5)
    # print(score)
