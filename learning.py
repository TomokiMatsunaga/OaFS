from tkinter import filedialog
from tkinter import *
import os
import argparse
import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import glob
import random
import tensorflow as tf
import csv
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from model import oafs_model
from evaluation import (instrument_list, multidataset_list, instfamily_conversion, framelevel_evaluate,
                        notelevel_evaluate, framestream_evaluate, notestream_evaluate, framelevel_evaluate_instwise,
                        notelevel_evaluate_instwise, midi_file_create)

# Parameter setting
dataset_list = ['Slakh']  # 'MusicNet', 'URMP', 'Slakh', 'GuitarSet', 'MAESTRO'
train_mode = 0  # 0 : train   1 : predict
evaluation_mode = 1  # 0 : off   1 : on
train_feature_path = ['MLCFP/Slakh/train/feature']
val_feature_path = ['MLCFP/Slakh/validation/feature']

timesteps = 256
feature_num = 256
ch_num = 9
inst_class = 34
epoch = 200
batchsize = 4
steps = 8000
valbatchsize = 4
valsteps = 2000
start = 0
end = 127
weight_decay = 0.1
dropout_rate = 0.1
earlystop = 30
initial_learning_rate = 1e-5
warmup_target = 1e-4
warmup_steps = 5

step_size = 240
inst_th = 0.99
onset_th = 6.0
frame_th = 2.5
onset_prominence = 5.0
dura_th = 0.06  # [s]
max_length = 0.08  # [s]
H = 0.02  # hop size [s]

###


def batchwise_feature(feature_files, timesteps=timesteps, start=start, end=end):
    feature_id = random.choice(feature_files)
    with h5py.File(feature_id, 'r') as feature_f:
        feature_mat = feature_f['feature']
        with h5py.File(os.path.dirname(os.path.dirname(feature_id)) + '/label/' +
                       os.path.basename(feature_id), 'r') as label_f:
            onset_label = label_f['onset_label']
            frame_label = label_f['frame_label']
            if len(feature_mat) > timesteps:
                start_id = random.choice(list(range(len(feature_mat) - timesteps + 1)))
                x = feature_mat[start_id:start_id + timesteps]
                y_onset = onset_label[start_id:start_id + timesteps, start:end + 1]
                y_frame = frame_label[start_id:start_id + timesteps, start:end + 1]
            elif len(feature_mat) == timesteps:
                x = feature_mat
                y_onset = onset_label[:, start:end + 1]
                y_frame = frame_label[:, start:end + 1]
            else:
                zero_pad = np.zeros((timesteps - len(feature_mat),) + np.shape(feature_mat)[1:])
                label_zero_pad = np.zeros((timesteps - len(onset_label), end - start + 1) + np.shape(onset_label)[2:])
                x = np.concatenate([feature_mat, zero_pad])
                y_onset = np.concatenate([onset_label[:, start:end + 1], label_zero_pad])
                y_frame = np.concatenate([frame_label[:, start:end + 1], label_zero_pad])
            y = (y_onset, y_frame)
    return x, y


def dataset_create(train_feature_files, val_feature_files, timesteps=timesteps, feature_num=feature_num, ch_num=ch_num,
                   out_class=inst_class + 1, epoch=epoch, batchsize=batchsize, steps=steps, valbatchsize=valbatchsize,
                   valsteps=valsteps, start=start, end=end):

    def train_gen():
        for _ in range(epoch * batchsize * steps):
            x, y = batchwise_feature(train_feature_files, timesteps, start, end)
            yield x, y

    train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature=(
        tf.TensorSpec(shape=(timesteps, feature_num, ch_num), dtype=tf.float32),
        (tf.TensorSpec(shape=(timesteps, end - start + 1, out_class), dtype=tf.float32),
         tf.TensorSpec(shape=(timesteps, end - start + 1, out_class), dtype=tf.float32)))
                                                   ).batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    def val_gen():
        for _ in range(epoch * valbatchsize * valsteps):
            x, y = batchwise_feature(val_feature_files, timesteps, start, end)
            yield x, y

    val_dataset = tf.data.Dataset.from_generator(val_gen, output_signature=(
        tf.TensorSpec(shape=(timesteps, feature_num, ch_num), dtype=tf.float32),
        (tf.TensorSpec(shape=(timesteps, end - start + 1, out_class), dtype=tf.float32),
         tf.TensorSpec(shape=(timesteps, end - start + 1, out_class), dtype=tf.float32)))
                                                 ).batch(valbatchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate=initial_learning_rate, warmup_target=warmup_target,
                 warmup_steps=warmup_steps):
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.warmup_target = tf.cast(warmup_target, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        if step < self.warmup_steps:
            return ((self.warmup_target - self.initial_learning_rate) / self.warmup_steps * step +
                    self.initial_learning_rate)
        else:
            return tf.constant(self.warmup_target)


def train(train_feature_path=['train/feature'], val_feature_path=['validation/feature'], timesteps=timesteps,
          feature_num=feature_num, ch_num=ch_num, inst_class=inst_class, epoch=epoch, batchsize=batchsize, steps=steps,
          valbatchsize=valbatchsize, valsteps=valsteps, start=start, end=end, weight_decay=weight_decay,
          dropout_rate=dropout_rate, earlystop=earlystop, initial_learning_rate=initial_learning_rate,
          warmup_target=warmup_target, warmup_steps=warmup_steps):
    train_feature_files = []
    for path_name in train_feature_path:
        train_feature_files.extend(glob.glob(path_name + '/*.hdf5'))
    val_feature_files = []
    for path_name in val_feature_path:
        val_feature_files.extend(glob.glob(path_name + '/*.hdf5'))
    train_dataset, val_dataset = dataset_create(train_feature_files, val_feature_files, timesteps=timesteps,
                                                feature_num=feature_num, ch_num=ch_num, out_class=inst_class + 1,
                                                epoch=epoch, batchsize=batchsize, steps=steps,
                                                valbatchsize=valbatchsize, valsteps=valsteps, start=start, end=end)
    model = oafs_model(feature_num=feature_num, timesteps=timesteps, ch_num=ch_num, out_class=inst_class + 1,
                       dropout_rate=dropout_rate, transformer_dropout_rate=dropout_rate, start=start, end=end)

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_learning_rate, weight_decay=weight_decay),
                  loss={'onset': tf.keras.losses.BinaryCrossentropy(axis=None),
                        'frame': tf.keras.losses.BinaryCrossentropy(axis=None)},
                  metrics={'onset': 'accuracy', 'frame': 'accuracy'}, loss_weights={'onset': 10.0, 'frame': 1.0})
    callbacks = [EarlyStopping(monitor='val_frame_accuracy', patience=earlystop),
                 ModelCheckpoint('checkpoints/{epoch:02d}-{loss:.4f}-{onset_accuracy:.4f}-{frame_accuracy:.4f}-'
                                 '{val_loss:.4f}-{val_onset_accuracy:.4f}-{val_frame_accuracy:.4f}'),
                 LearningRateScheduler(WarmUp(initial_learning_rate=initial_learning_rate, warmup_target=warmup_target,
                                              warmup_steps=warmup_steps))]
    model.fit(train_dataset, callbacks=callbacks, epochs=epoch, validation_data=val_dataset, steps_per_epoch=steps,
              validation_steps=valsteps)


def batchwise_predict(feature, loaded_model, timesteps=timesteps, step_size=step_size, batchsize=batchsize,
                      inst_class=inst_class, start=start, end=end):
    # frame reconstruction step
    batch = []
    for i in range(np.ceil((np.shape(feature)[0] - timesteps) / step_size).astype(int) + 1):
        if i * step_size + timesteps <= np.shape(feature)[0]:
            feat = feature[i * step_size:i * step_size + timesteps]
        else:
            feat = feature[i * step_size:]
            pad = np.zeros((i * step_size + timesteps - np.shape(feature)[0],) + np.shape(feature)[1:])
            feat = np.concatenate([feat, pad])
        batch.append(feat)
    batch = np.array(batch)
    batch_pred = loaded_model.predict(batch, batch_size=batchsize)
    batch_onset_pred = batch_pred[0]
    batch_frame_pred = batch_pred[1]
    onset_pred = np.zeros(np.shape(feature)[:1] + (end - start + 1, inst_class + 1))
    frame_pred = np.zeros(np.shape(feature)[:1] + (end - start + 1, inst_class + 1))
    mask = np.zeros_like(onset_pred)
    for i in range(np.shape(batch_onset_pred)[0]):
        if i * step_size + timesteps <= np.shape(feature)[0]:
            onset_pred[i * step_size:i * step_size + timesteps] += batch_onset_pred[i]
            frame_pred[i * step_size:i * step_size + timesteps] += batch_frame_pred[i]
            mask[i * step_size:i * step_size + timesteps] += 1
        else:
            onset_pred[i * step_size:] += batch_onset_pred[i][:np.shape(feature)[0] - i * step_size]
            frame_pred[i * step_size:] += batch_frame_pred[i][:np.shape(feature)[0] - i * step_size]
            mask[i * step_size:] += 1
    onset_pred /= mask
    frame_pred /= mask

    # global normalization step
    onset_pred = stats.zscore(onset_pred[:, :, :inst_class], None)
    frame_pred = stats.zscore(frame_pred[:, :, :inst_class], None)
    return onset_pred, frame_pred


def note_inference(onset_pred, frame_pred, start, end, inst_th, onset_th, frame_th, onset_prominence, dura_th,
                   max_length):
    frame = np.zeros((len(onset_pred), end - start + 1))
    note = np.empty((0, 3))

    # instrument selection step
    std = (np.std(onset_pred) + np.std(frame_pred)) / 2
    print('std:', std)
    if std < inst_th:
        return frame, note

    # local normalization step
    onset_zscore = stats.zscore(onset_pred, None)
    frame_zscore = stats.zscore(frame_pred, None)

    # note inference step
    frame_thres = np.where(frame_zscore < frame_th, 0, frame_zscore - frame_th)
    for ii in range(end - start + 1):
        if np.sum(frame_thres[:, ii]) == 0:
            continue
        peaks, _ = find_peaks(onset_zscore[:, ii], distance=dura_th, prominence=onset_prominence, height=onset_th)
        if len(peaks) == 0:
            continue
        for j in range(len(peaks)):
            if j == len(peaks) - 1:
                upper = np.shape(frame_thres)[0]
            else:
                upper = peaks[j + 1]
            for k in range(peaks[j], upper):
                if np.sum(frame_thres[k:k + max_length, ii]) == 0:
                    if k == peaks[j]:
                        break
                    else:
                        frame[peaks[j]:k, ii] = 1
                        note = np.append(note, np.array([[peaks[j], k - 1, ii + start]]), axis=0)
                        break
            else:
                frame[peaks[j]:upper, ii] = 1
                note = np.append(note, np.array([[peaks[j], upper - 1, ii + start]]), axis=0)
    return frame, note


def note_select(onset_pred, frame_pred, instlist, H=H, inst_class=inst_class, start=start, end=end, inst_th=inst_th,
                onset_th=onset_th, frame_th=frame_th, onset_prominence=onset_prominence, dura_th=dura_th,
                max_length=max_length):
    dura_th = round(dura_th / H)
    max_length = round(max_length / H)

    onset_pred_mix = np.max(onset_pred, axis=-1)
    frame_pred_mix = np.max(frame_pred, axis=-1)
    frame, note = note_inference(onset_pred_mix, frame_pred_mix, start, end, inst_th, onset_th, frame_th,
                                 onset_prominence, dura_th, max_length)
    note[:, :2] *= H
    note = note[np.argsort(note[:, 0])]

    framestream = np.zeros_like(onset_pred)
    notestream = np.empty((0, 4))
    for i in range(inst_class):
        onset_pred_inst = onset_pred[:, :, i]
        frame_pred_inst = frame_pred[:, :, i]
        frame_inst, note_inst = note_inference(onset_pred_inst, frame_pred_inst, start, end, inst_th, onset_th,
                                               frame_th, onset_prominence, dura_th, max_length)
        if len(note_inst) == 0:
            continue
        framestream[:, :, i] = frame_inst
        notestream = np.append(notestream, np.hstack([note_inst, np.full((len(note_inst), 1), instlist[i])]),
                               axis=0)
    notestream[:, :2] *= H
    notestream = notestream[np.argsort(notestream[:, 0])]
    return frame, note, framestream, notestream


def transcribe(feature, checkpoint_filepath, instlist, timesteps=timesteps, step_size=step_size, batchsize=batchsize,
               H=H, inst_class=inst_class, start=start, end=end, inst_th=inst_th, onset_th=onset_th, frame_th=frame_th,
               onset_prominence=onset_prominence, dura_th=dura_th, max_length=max_length):
    loaded_model = tf.keras.models.load_model(checkpoint_filepath)
    onset_pred, frame_pred = batchwise_predict(
        feature, loaded_model, timesteps=timesteps, step_size=step_size, batchsize=batchsize, inst_class=inst_class,
        start=start, end=end)
    frame, note, framestream, notestream = note_select(
        onset_pred, frame_pred, instlist, H=H, inst_class=inst_class, start=start, end=end, inst_th=inst_th,
        onset_th=onset_th, frame_th=frame_th, onset_prominence=onset_prominence, dura_th=dura_th, max_length=max_length)
    return frame, note, framestream, notestream


def onset_and_frame_streams_evaluate(feature_folder, checkpoint_folder, csv_path, dataset_list, midi_path, H=H):
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

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'inst_TP', 'inst_FP', 'inst_FN', 'frame-level_TP', 'frame-level_FP', 'frame-level_FN',
                         'note-level_TP', 'note-level_FP', 'note-level_FN', 'frame-stream_TP', 'frame-stream_FP',
                         'frame-stream_FN', 'note-stream_TP', 'note-stream_FP', 'note-stream_FN'])

    csv_path_instwise = os.path.splitext(csv_path)[0] + '_instwise.csv'
    inst_name = multidataset_list(dataset_list)
    inst_name = [str(n) for n in inst_name]
    item_name = ['name']
    for i in inst_name:
        item_name.extend(list(map(lambda x, y: x + y, [i] * 3, ['_frame_TP', '_frame_FP', '_frame_FN'])))
    for i in inst_name:
        item_name.extend(list(map(lambda x, y: x + y, [i] * 3, ['_note_TP', '_note_FP', '_note_FN'])))
    with open(csv_path_instwise, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item_name)

    csv_path_family = os.path.splitext(csv_path)[0] + '_family.csv'
    with open(csv_path_family, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'inst_TP', 'inst_FP', 'inst_FN', 'frame-level_TP', 'frame-level_FP', 'frame-level_FN',
                         'note-level_TP', 'note-level_FP', 'note-level_FN', 'frame-stream_TP', 'frame-stream_FP',
                         'frame-stream_FN', 'note-stream_TP', 'note-stream_FP', 'note-stream_FN'])

    csv_path_family_instwise = os.path.splitext(csv_path)[0] + '_family_instwise.csv'
    inst_name = instrument_list('InstFamily')
    inst_name = [str(n) for n in inst_name]
    item_name = ['name']
    for i in inst_name:
        item_name.extend(list(map(lambda x, y: x + y, [i] * 3, ['_frame_TP', '_frame_FP', '_frame_FN'])))
    for i in inst_name:
        item_name.extend(list(map(lambda x, y: x + y, [i] * 3, ['_note_TP', '_note_FP', '_note_FN'])))
    with open(csv_path_family_instwise, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item_name)

    for feature_file in glob.glob(feature_folder):
        with h5py.File(feature_file, 'r') as feature_f:
            feature = feature_f['feature'][...]
        instlist = multidataset_list(dataset_list)
        inst_class = len(instlist)
        frame, note, framestream, notestream = transcribe(feature, checkpoint_folder, instlist, inst_class=inst_class)
        frame = np.pad(frame, ((0, 0), (start, 127 - end)))
        framestream = np.pad(framestream, ((0, 0), (start, 127 - end), (0, 0)))
        label_file = os.path.dirname(os.path.dirname(feature_file)) + '/label/' + os.path.basename(feature_file)
        with h5py.File(label_file, 'r') as label_f:
            onset_label = label_f['onset_label'][...]
            frame_label = label_f['frame_label'][...]
            note_label = label_f['note_label'][...]

        midi_file_create(notestream, midi_path + '/' + os.path.splitext(os.path.basename(feature_file))[0])
        instnum = np.unique(notestream[:, 3]).astype(int)
        instnum_label = np.unique(note_label[:, 3]).astype(int)
        inst_TP = len(np.intersect1d(instnum, instnum_label))
        inst_FP = len(instnum) - inst_TP
        inst_FN = len(instnum_label) - inst_TP
        print('inst-detection', 'TP:', inst_TP, 'FP:', inst_FP, 'FN:', inst_FN)
        print('F-measure:', 2 * inst_TP / (2 * inst_TP + inst_FP + inst_FN))
        # feval = framelevel_evaluate(frame, frame_label[:, :, :-1], instlist)
        # neval = notelevel_evaluate(note, note_label, percussive_on=True)
        feval = framelevel_evaluate(framestream, frame_label[:, :, :-1], instlist)
        print('frame-level', 'TP:', feval[0], 'FP:', feval[1], 'FN:', feval[2])
        print('F-measure:', 2 * feval[0] / (2 * feval[0] + feval[1] + feval[2]))
        neval = notelevel_evaluate(notestream, note_label, percussive_on=True)
        print('note-level', 'TP:', neval[0], 'FP:', neval[1], 'FN:', neval[2])
        print('F-measure:', 2 * neval[0] / (2 * neval[0] + neval[1] + neval[2]))
        feval_stream = framestream_evaluate(framestream, frame_label[:, :, :-1], instlist)
        print('frame-stream', 'TP:', feval_stream[0], 'FP:', feval_stream[1], 'FN:', feval_stream[2])
        print('F-measure:', 2 * feval_stream[0] / (2 * feval_stream[0] + feval_stream[1] + feval_stream[2]))
        neval_stream = notestream_evaluate(notestream, note_label, instlist, percussive_on=True)
        print('note-stream', 'TP:', neval_stream[0], 'FP:', neval_stream[1], 'FN:', neval_stream[2])
        print('F-measure:', 2 * neval_stream[0] / (2 * neval_stream[0] + neval_stream[1] + neval_stream[2]))
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([os.path.splitext(os.path.basename(feature_file))[0], inst_TP, inst_FP, inst_FN, feval[0],
                             feval[1], feval[2], neval[0], neval[1], neval[2], feval_stream[0], feval_stream[1],
                             feval_stream[2], neval_stream[0], neval_stream[1], neval_stream[2]])

        feval_instwise = framelevel_evaluate_instwise(framestream, frame_label[:, :, :-1])
        neval_instwise = notelevel_evaluate_instwise(notestream, note_label, instlist)
        file_name = [os.path.splitext(os.path.basename(feature_file))[0]]
        file_name.extend(feval_instwise.ravel())
        file_name.extend(neval_instwise.ravel())
        with open(csv_path_instwise, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(file_name)

        instlist_family = instrument_list('InstFamily')
        frame_family, note_family = instfamily_conversion(framestream, notestream, instlist_family, H)
        frame_label_family, note_label_family = instfamily_conversion(frame_label, note_label, instlist_family, H)
        instnum_family = np.unique(note_family[:, 3]).astype(int)
        instnum_label_family = np.unique(note_label_family[:, 3]).astype(int)
        instfamily_TP = len(np.intersect1d(instnum_family, instnum_label_family))
        instfamily_FP = len(instnum_family) - instfamily_TP
        instfamily_FN = len(instnum_label_family) - instfamily_TP
        feval_family = framelevel_evaluate(frame_family, frame_label_family, instlist_family)
        neval_family = notelevel_evaluate(note_family, note_label_family, percussive_on=True)
        feval_stream_family = framestream_evaluate(frame_family, frame_label_family, instlist_family)
        neval_stream_family = notestream_evaluate(note_family, note_label_family, instlist_family, percussive_on=True)
        with open(csv_path_family, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([os.path.splitext(os.path.basename(feature_file))[0], instfamily_TP, instfamily_FP,
                             instfamily_FN, feval_family[0], feval_family[1], feval_family[2], neval_family[0],
                             neval_family[1], neval_family[2], feval_stream_family[0], feval_stream_family[1],
                             feval_stream_family[2], neval_stream_family[0], neval_stream_family[1],
                             neval_stream_family[2]])

        feval_family_instwise = framelevel_evaluate_instwise(frame_family, frame_label_family)
        neval_family_instwise = notelevel_evaluate_instwise(note_family, note_label_family, instlist_family)
        file_name = [os.path.splitext(os.path.basename(feature_file))[0]]
        file_name.extend(feval_family_instwise.ravel())
        file_name.extend(neval_family_instwise.ravel())
        with open(csv_path_family_instwise, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_list', nargs='+')
    parser.add_argument('--train_mode', type=int)
    parser.add_argument('--evaluation_mode', type=int)
    parser.add_argument('--train_feature_path', nargs='+')
    parser.add_argument('--val_feature_path', nargs='+')
    args = parser.parse_args()
    if args.dataset_list is not None:
        dataset_list = args.dataset_list
    if args.train_mode is not None:
        if args.train_mode != 0 and args.train_mode != 1:
            raise Exception('Select 0 or 1 for train_mode')
        else:
            train_mode = args.train_mode
    if args.evaluation_mode is not None:
        if args.evaluation_mode != 0 and args.evaluation_mode != 1:
            raise Exception('Select 0 or 1 for evaluation_mode')
        else:
            evaluation_mode = args.evaluation_mode
    if args.train_feature_path is not None:
        train_feature_path = args.train_feature_path
    if args.val_feature_path is not None:
        val_feature_path = args.val_feature_path

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
    if train_mode == 0:
        os.makedirs('checkpoints', exist_ok=True)
        train(train_feature_path=train_feature_path, val_feature_path=val_feature_path)
    elif train_mode == 1:
        root = Tk()
        root.withdraw()
        print("Please open feature files")
        feature_file = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5")],
                                                  initialdir='MLCFP')
        print(feature_file)
        if evaluation_mode == 1:
            print("Please open label files")
            label_file = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5")],
                                                    initialdir='MLCFP')
            print(label_file)
        print("Please open checkpoint folder")
        checkpoint_folder = filedialog.askdirectory(initialdir='checkpoints')
        print(checkpoint_folder)
        with h5py.File(feature_file, 'r') as feature_f:
            feature = feature_f['feature'][...]
        instlist = multidataset_list(dataset_list)
        frame, note, framestream, notestream = transcribe(feature, checkpoint_folder, instlist)
        frame = np.pad(frame, ((0, 0), (start, 127 - end)))
        framestream = np.pad(framestream, ((0, 0), (start, 127 - end), (0, 0)))
        midi_file_create(notestream, 'temp/transcription')
        if evaluation_mode == 0:
            np.savez_compressed('temp/pianoroll', note_est=notestream)

        if evaluation_mode == 1:
            with h5py.File(label_file, 'r') as label_f:
                onset_label = label_f['onset_label'][...]
                frame_label = label_f['frame_label'][...]
                note_label = label_f['note_label'][...]
            np.savez_compressed('temp/pianoroll', note_est=notestream, note_ref=note_label)

            feval = framelevel_evaluate(framestream, frame_label[:, :, :-1], instlist)
            print('frame-level', 'TP:', feval[0], 'FP:', feval[1], 'FN:', feval[2])
            print('F-measure:', 2 * feval[0] / (2 * feval[0] + feval[1] + feval[2]))
            neval = notelevel_evaluate(notestream, note_label, percussive_on=True)
            print('note-level', 'TP:', neval[0], 'FP:', neval[1], 'FN:', neval[2])
            print('F-measure:', 2 * neval[0] / (2 * neval[0] + neval[1] + neval[2]))
            feval_stream = framestream_evaluate(framestream, frame_label[:, :, :-1], instlist)
            print('frame-stream', 'TP:', feval_stream[0], 'FP:', feval_stream[1], 'FN:', feval_stream[2])
            print('F-measure:', 2 * feval_stream[0] / (2 * feval_stream[0] + feval_stream[1] + feval_stream[2]))
            neval_stream = notestream_evaluate(notestream, note_label, instlist, percussive_on=True)
            print('note-stream', 'TP:', neval_stream[0], 'FP:', neval_stream[1], 'FN:', neval_stream[2])
            print('F-measure:', 2 * neval_stream[0] / (2 * neval_stream[0] + neval_stream[1] + neval_stream[2]))
            feval_instwise = framelevel_evaluate_instwise(framestream, frame_label[:, :, :-1])
            for k in range(len(instlist)):
                if np.sum(feval_instwise[k]) == 0:
                    continue
                print('frame-level (inst:' + str(instlist[k]) + ')', 'TP:', feval_instwise[k][0], 'FP:',
                      feval_instwise[k][1], 'FN:', feval_instwise[k][2])
                print('F-measure:', 2 * feval_instwise[k][0] / (2 * feval_instwise[k][0] + feval_instwise[k][1]
                                                                + feval_instwise[k][2]))
            neval_instwise = notelevel_evaluate_instwise(notestream, note_label, instlist)
            for k in range(len(instlist)):
                if np.sum(neval_instwise[k]) == 0:
                    continue
                print('note-level (inst:' + str(instlist[k]) + ')', 'TP:', neval_instwise[k][0], 'FP:',
                      neval_instwise[k][1], 'FN:', neval_instwise[k][2])
                print('F-measure:', 2 * neval_instwise[k][0] / (2 * neval_instwise[k][0] + neval_instwise[k][1]
                                                                + neval_instwise[k][2]))
