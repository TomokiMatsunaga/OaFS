from MLCFP import parallel_extract_mlcfp, parallel_evaluation, extraction_detail_eval
from evaluation import instrument_list, multidataset_list, parallel_label_conversion, parallel_label_instfamily
import pandas as pd

dataset = 'Slakh'  # 'MusicNet', 'URMP', 'Slakh', 'GuitarSet', 'MAESTRO'
file_path = 'Datasets'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--file_path')
    args = parser.parse_args()
    if args.dataset is not None:
        dataset = args.dataset
    if args.file_path is not None:
        file_path = args.file_path

    if dataset == 'Slakh':
        parallel_extract_mlcfp(file_path + '/slakh2100_redux/train/*/*.wav',
                               file_path + '/slakh2100_redux/',
                               'MLCFP/Slakh/train', 'Slakh')
        parallel_extract_mlcfp(file_path + '/slakh2100_redux/validation/*/*.wav',
                               file_path + '/slakh2100_redux/',
                               'MLCFP/Slakh/validation', 'Slakh')
        parallel_extract_mlcfp(file_path + '/slakh2100_redux/test/*/*.wav',
                               file_path + '/slakh2100_redux/',
                               'MLCFP/Slakh/test', 'Slakh')
    elif dataset == 'URMP':
        parallel_extract_mlcfp(file_path + '/URMP/Dataset/*/AuMix*.wav',
                               file_path + '/URMP_label/',
                               'MLCFP/URMP/', 'URMP')
    elif dataset == 'GuitarSet':
        parallel_extract_mlcfp(file_path + 'GuitarSet/audio_hex-pickup_original/*.wav',
                               file_path + 'GuitarSet_label/',
                               'MLCFP/GuitarSet/', 'GuitarSet')
    elif dataset == 'MusicNet':
        parallel_extract_mlcfp(file_path + 'musicnet/train_data/*.wav',
                               file_path + 'musicnet/train_labels/',
                               'MLCFP/MusicNet/', 'MusicNet')
        parallel_extract_mlcfp(file_path + 'musicnet/test_data/*.wav',
                               file_path + 'musicnet/test_labels/',
                               'MLCFP/MusicNet/', 'MusicNet')
    elif dataset == 'MAESTRO':
        parallel_extract_mlcfp(file_path + 'maestro-v3.0.0/*/*.wav',
                               file_path + 'maestro-v3.0.0/',
                               'MLCFP/MAESTRO/', 'MAESTRO')
    else:
        raise Exception('The specified dataset is not registered')
