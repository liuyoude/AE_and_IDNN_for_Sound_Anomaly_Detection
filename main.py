from train import *
from test import *
from pre_data import *
from utils import *

def train(param):
    os.makedirs(param['model_dir'], exist_ok=True)
    dirs = utils.select_dirs(param)
    process_mt_list = param['process_mt_list']
    for target_dir in dirs:
        machine_type = os.path.split(target_dir)[-1]
        if machine_type not in process_mt_list:
            continue
        pre_data_path = param['pre_data_dir'] + f'/{machine_type}.db'
        print(f'loading dataset [{machine_type}] ......')
        my_dataset = dataset.MyDataset(pre_data_path, keys=['log_mel'])
        train_loader = DataLoader(my_dataset, batch_size=param['batch_size'], shuffle=True)
        print('training ......')
        train(train_loader, machine_type, model_name=param['model_name'])

def test(param):
    process_mt_list = param['process_mt_list']
    pool_type = param['pool_type']
    os.makedirs(param['result_dir'], exist_ok=True)
    result_dir = param['result_dir']
    dirs = utils.select_dirs(param)
    csv_lines = []
    for pt in pool_type:
        for idx, target_dir in enumerate(dirs):
            machine_type = os.path.split(target_dir)[-1]
            if machine_type not in process_mt_list:
                continue
            print('\n' + '=' * 20)
            print(f'[{idx + 1}/{len(dirs)}] {target_dir} [{pt}]')
            evalute_test_machine(param['model_name'], target_dir, pool_type=pt)
        result_file_name = param['result_file']
        result_path = f'{result_dir}/{result_file_name}'
    utils.save_csv(save_file_path=result_path, save_data=csv_lines)

def main():
    # load yaml config file
    param = load_yaml(file_name='./config.yaml')
    # preprocess dataset
    save_pre_data_file(param,
                       n_mels=param['n_mels'],
                       frames=param['frames'],
                       n_fft=param['n_fft'],
                       hop_length=param['hop_length'],
                       power=param['power'])
    # train
    train(param)

    # test
    test(param)