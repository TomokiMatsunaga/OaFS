from tkinter import filedialog
from tkinter import *
import os
import argparse
import tensorflow as tf
from MLCFP import extract_feature_low_memory
from learning import transcribe
from evaluation import multidataset_list, midi_file_create

dataset_list = ['Slakh']  # 'MusicNet', 'URMP', 'Slakh', 'GuitarSet', 'MAESTRO'

os.makedirs('temp/transcription', exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_list', nargs='+')
    args = parser.parse_args()
    if args.dataset_list is not None:
        dataset_list = args.dataset_list
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print('Please open .wav files')
    root = Tk()
    root.withdraw()
    wav_file = filedialog.askopenfilename(filetypes=[('.wav files', '*.wav')])
    print(wav_file)
    print("Please open checkpoint folder")
    checkpoint_folder = filedialog.askdirectory(initialdir='checkpoints')
    print(checkpoint_folder)
    feature, t = extract_feature_low_memory(wav_file, output_path='', dataset='', save_feature=False)
    instlist = multidataset_list(dataset_list)
    frame, note, framestream, notestream = transcribe(feature, checkpoint_folder, instlist)
    midi_file_create(notestream, 'temp/transcription/' + os.path.splitext(os.path.basename(wav_file))[0])

