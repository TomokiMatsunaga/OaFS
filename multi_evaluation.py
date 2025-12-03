from tkinter import filedialog
from tkinter import *
import os
import argparse
import pandas as pd
import numpy as np
from learning import onset_and_frame_streams_evaluate

feature_folder = 'test/feature/*.hdf5'
csv_path = 'temp/Slakh_eval.csv'
dataset_list = ['Slakh']
midi_path = 'MIDI/Slakh'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_folder')
    parser.add_argument('--csv_path')
    parser.add_argument('--dataset_list', nargs='+')
    parser.add_argument('--midi_path')
    args = parser.parse_args()
    if args.feature_folder is not None:
        feature_folder = args.feature_folder
    if args.csv_path is not None:
        csv_path = args.csv_path
    if args.dataset_list is not None:
        dataset_list = args.dataset_list
    if args.midi_path is not None:
        midi_path = args.midi_path

    os.makedirs(midi_path, exist_ok=True)

    root = Tk()
    root.withdraw()
    print("Please open checkpoint folder")
    checkpoint_folder = filedialog.askdirectory(initialdir='checkpoints')
    print(checkpoint_folder)
    onset_and_frame_streams_evaluate(feature_folder, checkpoint_folder, csv_path, dataset_list, midi_path)

    evaluation_list = pd.read_csv(csv_path)
    TP = np.sum(evaluation_list['frame-level_TP'].values)
    FP = np.sum(evaluation_list['frame-level_FP'].values)
    FN = np.sum(evaluation_list['frame-level_FN'].values)
    frame_level = 2 * TP / (2 * TP + FP + FN) * 100
    TP = np.sum(evaluation_list['note-level_TP'].values)
    FP = np.sum(evaluation_list['note-level_FP'].values)
    FN = np.sum(evaluation_list['note-level_FN'].values)
    note_level = 2 * TP / (2 * TP + FP + FN) * 100
    TP = np.sum(evaluation_list['frame-stream_TP'].values)
    FP = np.sum(evaluation_list['frame-stream_FP'].values)
    FN = np.sum(evaluation_list['frame-stream_FN'].values)
    frame_stream = 2 * TP / (2 * TP + FP + FN) * 100
    TP = np.sum(evaluation_list['note-stream_TP'].values)
    FP = np.sum(evaluation_list['note-stream_FP'].values)
    FN = np.sum(evaluation_list['note-stream_FN'].values)
    note_stream = 2 * TP / (2 * TP + FP + FN) * 100
    TP = np.sum(evaluation_list['inst_TP'].values)
    FP = np.sum(evaluation_list['inst_FP'].values)
    FN = np.sum(evaluation_list['inst_FN'].values)
    inst = 2 * TP / (2 * TP + FP + FN) * 100
    print(checkpoint_folder)
    print('frame_level:', frame_level)
    print('note_level:', note_level)
    print('frame_stream:', frame_stream)
    print('note_stream:', note_stream)
    print('inst:', inst)

