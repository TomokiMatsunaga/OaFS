import numpy as np
from scipy.io import loadmat
import pandas as pd
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import glob
# import jams
import os
import h5py
from concurrent.futures import ProcessPoolExecutor

accrange = 0.05  # the range of a reference onset [s]


def instrument_list(dataset):
    if dataset == 'MAESTRO':
        instlist = [0]
    elif dataset == 'GuitarSet':
        instlist = [25]
    elif dataset == 'MusicNet':
        instlist = [0, 6, 40, 41, 42, 43, 60, 68, 70, 71, 73]
    elif dataset == 'URMP':
        instlist = [40, 41, 42, 43, 56, 57, 58, 60, 65, 68, 70, 71, 73]
    elif dataset == 'Slakh':
        instlist = [0, 4, 8, 16, 25, 26, 29, 32, 33, 40, 41, 42, 43, 46, 47, 48, 50, 52, 56, 57, 58, 60, 61, 65, 66,
                    67, 68, 69, 70, 71, 73, 80, 88, 128]
    elif dataset == 'InstFamily':
        instlist = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 128]
    else:
        raise Exception('Select the appropriate value for dataset')
    return instlist


def midi_label_create(label_file):
    mid = mido.MidiFile(label_file)
    mididict = []
    onset = np.empty((0, 4))
    offset = np.empty((0, 4))
    for x in mid:
        mididict.append(x.dict())
    program_list = np.zeros((16, 2), dtype=int)
    program_list[:, 0] = np.arange(16)
    program_list[:, 1] = -1
    mem = 0
    for y in mididict:
        settime = y['time'] + mem
        y['time'] = settime
        mem = y['time']
        if y['type'] == 'program_change':
            program_list[y['channel'], 1] = y['program']
        if y['type'] == 'note_on' or y['type'] == 'note_off':
            if y['channel'] == 9:
                program = 128
            else:
                program = program_list[y['channel'], 1]
                if program == -1:
                    program = 128
            if y['type'] == 'note_on' and y['velocity'] == 0:
                y['type'] = 'note_off'
            if y['type'] == 'note_on':
                onset = np.append(onset, np.array([[y['time'], y['note'], program, y['channel']]]), axis=0)
            elif y['type'] == 'note_off':
                offset = np.append(offset, np.array([[y['time'], y['note'], program, y['channel']]]), axis=0)
    note_label = np.empty((0, 4))
    d = np.zeros(np.shape(offset)[0])
    for i in range(np.shape(onset)[0]):
        matchvec = np.where(np.all(offset[:, 1:] == onset[i, 1:], 1))[0]
        matchonvec = np.where(np.all(onset == onset[i], 1))[0]
        if np.any(matchonvec > i):
            continue
        if np.all(d[matchvec] != 0):
            continue
        imatch = matchvec[d[matchvec] == 0][0]
        while onset[i, 0] > offset[imatch, 0]:
            d[imatch] = 1
            if np.all(d[matchvec] != 0):
                break
            imatch = matchvec[d[matchvec] == 0][0]
        else:
            note_label = np.append(
                note_label, np.array([[onset[i, 0], offset[imatch, 0], onset[i, 1], onset[i, 2]]]), axis=0)
            d[imatch] = 1
    return note_label


def note_label_create(label_file, dataset, augmentation, length=0):
    if dataset == 'MAPS':
        df = pd.read_table(label_file)
        note_label = df.values.astype(float)
        note_label = np.append(note_label, np.zeros((1, np.shape(note_label)[0])).T, axis=1)
    # elif dataset == 'GuitarSet':
    #     jam = jams.load(label_file)
    #     annos = jam.search(namespace='note_midi')
    #     if len(annos) == 0:
    #         annos = jam.search(namespace='pitch_midi')
    #     note_label = np.empty((0, 4))
    #     for anno in annos:
    #         for note in anno:
    #             note_label = np.append(note_label, np.array([[note.time, note.time + note.duration,
    #                                                           int(round(note.value)), 25]]), axis=0)
    #     note_label = note_label[np.argsort(note_label[:, 0])]
    elif dataset == 'MusicNet':
        df = pd.read_csv(label_file)
        note_label = df.values[:, [0, 1, 3, 2]].astype(float)
        note_label[:, :2] /= 44100
        note_label[:, 3] -= 1
    elif dataset == 'Bach10':
        dic = loadmat(label_file, simplify_cells=True)
        note_dic = dic['GTNotes']
        note_label = np.empty((0, 4))
        inst = [40, 71, 65, 70]
        for x in range(len(note_dic)):
            for y in range(len(note_dic[x])):
                note_label = np.append(note_label, np.array([[note_dic[x][y][0][0], note_dic[x][y][0][-1],
                                                              round(note_dic[x][y][1][0]), inst[x]]]), axis=0)
        note_label[:, :2] *= 0.01
        note_label = note_label[np.argsort(note_label[:, 0])]
    elif dataset == 'TRIOS' or dataset == 'URMP' or dataset == 'GuitarSet':
        note_label = np.load(label_file)
    elif dataset == 'RWC' or dataset == 'MAESTRO':
        note_label = midi_label_create(label_file)
    elif dataset == 'Slakh':
        note_label = np.empty((0, 4))
        if augmentation:
            for mid_file in label_file:
                stems_note_label = midi_label_create(mid_file)
                note_label = np.append(note_label, stems_note_label, axis=0)
        else:
            for mid_file in glob.glob(label_file):
                stems_note_label = midi_label_create(mid_file)
                note_label = np.append(note_label, stems_note_label, axis=0)
        note_label = note_label[np.argsort(note_label[:, 0])]
        note_label[:, 3][(note_label[:, 3] == 1) | (note_label[:, 3] == 3) | (note_label[:, 3] == 6)] = 0
        note_label[:, 3][(note_label[:, 3] == 2) | (note_label[:, 3] == 5) | (note_label[:, 3] == 7)] = 4
        note_label[:, 3][(note_label[:, 3] == 9) | (note_label[:, 3] == 10) | (note_label[:, 3] == 11) |
                         (note_label[:, 3] == 12) | (note_label[:, 3] == 13) | (note_label[:, 3] == 14) |
                         (note_label[:, 3] == 15)] = 8
        note_label[:, 3][(note_label[:, 3] == 17) | (note_label[:, 3] == 18) | (note_label[:, 3] == 19) |
                         (note_label[:, 3] == 20) | (note_label[:, 3] == 21) | (note_label[:, 3] == 22) |
                         (note_label[:, 3] == 23)] = 16
        note_label[:, 3][note_label[:, 3] == 24] = 25
        note_label[:, 3][(note_label[:, 3] == 27) | (note_label[:, 3] == 28) | (note_label[:, 3] == 31)] = 26
        note_label[:, 3][note_label[:, 3] == 30] = 29
        note_label[:, 3][(note_label[:, 3] == 34) | (note_label[:, 3] == 35) | (note_label[:, 3] == 36) |
                         (note_label[:, 3] == 37)] = 33
        note_label[:, 3][(note_label[:, 3] == 44) | (note_label[:, 3] == 45) | (note_label[:, 3] == 49)] = 48
        note_label[:, 3][note_label[:, 3] == 51] = 50
        note_label[:, 3][(note_label[:, 3] == 53) | (note_label[:, 3] == 54)] = 52
        note_label[:, 3][note_label[:, 3] == 59] = 56
        note_label[:, 3][note_label[:, 3] == 64] = 65
        note_label[:, 3][(note_label[:, 3] == 72) | (note_label[:, 3] == 74) | (note_label[:, 3] == 75) |
                         (note_label[:, 3] == 76) | (note_label[:, 3] == 77) | (note_label[:, 3] == 78) |
                         (note_label[:, 3] == 79)] = 73
        note_label[:, 3][(note_label[:, 3] == 81) | (note_label[:, 3] == 82) | (note_label[:, 3] == 83) |
                         (note_label[:, 3] == 84) | (note_label[:, 3] == 85) | (note_label[:, 3] == 86) |
                         (note_label[:, 3] == 87)] = 80
        note_label[:, 3][(note_label[:, 3] == 89) | (note_label[:, 3] == 90) | (note_label[:, 3] == 91) |
                         (note_label[:, 3] == 92) | (note_label[:, 3] == 93) | (note_label[:, 3] == 94) |
                         (note_label[:, 3] == 95)] = 88
    else:
        raise Exception('Select the appropriate value for dataset')

    if length > 0:
        note_label = note_label[note_label[:, 0] < length]
        note_label[:, 1][note_label[:, 1] > length] = length
    return note_label


def label_create(label_file, dataset, t, length=0, training=False, augmentation=False):
    note_label = note_label_create(label_file, dataset, augmentation, length=length)
    if training:
        instlist = instrument_list(dataset)
        onset_label = np.zeros((len(t), 128, len(instlist) + 1))
        frame_label = np.zeros((len(t), 128, len(instlist) + 1))
        for k in range(len(instlist)):
            if np.any(note_label[:, 3] == instlist[k]):
                note_label_inst = note_label[note_label[:, 3] == instlist[k]]
                p_label = np.asarray(note_label_inst[:, 2], dtype=int)
                for i in range(len(note_label_inst)):
                    onset = np.where(t >= note_label_inst[i, 0])[0][0]
                    offset = np.where(t <= note_label_inst[i, 1])[0][-1]
                    onset_label[onset, p_label[i], k] = 1
                    if onset > offset:
                        frame_label[onset, p_label[i], k] = 1
                    else:
                        frame_label[onset:offset + 1, p_label[i], k] = 1
        onset_label[:, :, -1] = np.where(np.max(onset_label[:, :, :-1], 2) > 0, 0, 1)
        frame_label[:, :, -1] = np.where(np.max(frame_label[:, :, :-1], 2) > 0, 0, 1)
        return onset_label, frame_label, note_label
    else:
        instnum = np.unique(note_label[:, 3])
        frame_label = np.zeros((len(t), 128, len(instnum)), dtype=np.int64)
        for k in range(len(instnum)):
            note_label_inst = note_label[note_label[:, 3] == instnum[k]]
            p_label = np.asarray(note_label_inst[:, 2], dtype=int)
            for i in range(len(note_label_inst)):
                onset = np.where(t >= note_label_inst[i, 0])[0][0]
                offset = np.where(t <= note_label_inst[i, 1])[0][-1]
                if onset > offset:
                    frame_label[onset, p_label[i], k] = 1
                else:
                    frame_label[onset:offset + 1, p_label[i], k] = 1
        return frame_label, note_label


def multidataset_list(dataset_list):
    instlist = []
    for dataset in dataset_list:
        instlist_wise = instrument_list(dataset)
        instlist = np.append(instlist, instlist_wise)
        instlist = np.unique(instlist)
    return instlist


def label_conversion(output_path, label_file, instlist_org, instlist):
    with h5py.File(label_file, 'r') as label_f:
        onset_label_org = label_f['onset_label'][...]
        frame_label_org = label_f['frame_label'][...]
        note_label = label_f['note_label'][...]
    # note_label[:, 3][note_label[:, 3] == 6] = 0  # for MusicNet
    onset_label = np.zeros((len(onset_label_org), np.shape(onset_label_org)[1], len(instlist) + 1))
    frame_label = np.zeros((len(frame_label_org), np.shape(frame_label_org)[1], len(instlist) + 1))
    instlist = np.array(instlist)
    for k in range(len(instlist_org)):
        # if instlist_org[k] == 6:  # for MusicNet
        #     if np.any(onset_label[:, :, 0] == 1):
        #         continue
        #     else:
        #         inst_idx = np.where(instlist == 0)[0][0]
        #         onset_label[:, :, inst_idx] = onset_label_org[:, :, k]
        #         frame_label[:, :, inst_idx] = frame_label_org[:, :, k]
        # else:
        if np.any(instlist == instlist_org[k]):
            inst_idx = np.where(instlist == instlist_org[k])[0][0]
            onset_label[:, :, inst_idx] = onset_label_org[:, :, k]
            frame_label[:, :, inst_idx] = frame_label_org[:, :, k]
    onset_label[:, :, -1] = onset_label_org[:, :, -1]
    frame_label[:, :, -1] = frame_label_org[:, :, -1]
    with h5py.File(output_path + os.path.basename(label_file), 'w') as f:
        f.create_dataset('onset_label', data=onset_label, chunks=(128, 128, np.shape(onset_label)[-1]),
                         compression='gzip', shuffle=True, fletcher32=True)
        f.create_dataset('frame_label', data=frame_label, chunks=(128, 128, np.shape(frame_label)[-1]),
                         compression='gzip', shuffle=True, fletcher32=True)
        f.create_dataset('note_label', data=note_label, compression='gzip', shuffle=True, fletcher32=True)


def parallel_label_conversion(filename, instlist_org, instlist):
    output_path = os.path.dirname(os.path.dirname(filename)) + '/label-conversion/'
    os.makedirs(output_path, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        for label_file in glob.glob(filename):
            executor.submit(label_conversion, output_path, label_file, instlist_org, instlist)


def instfamily_conversion(frame_label_org, note_label, instlist, H):
    note_label[:, 3][(note_label[:, 3] >= 0) & (note_label[:, 3] <= 7)] = 0
    note_label[:, 3][(note_label[:, 3] >= 8) & (note_label[:, 3] <= 15)] = 8
    note_label[:, 3][(note_label[:, 3] >= 16) & (note_label[:, 3] <= 23)] = 16
    note_label[:, 3][(note_label[:, 3] >= 24) & (note_label[:, 3] <= 31)] = 24
    note_label[:, 3][(note_label[:, 3] >= 32) & (note_label[:, 3] <= 39)] = 32
    note_label[:, 3][(note_label[:, 3] >= 40) & (note_label[:, 3] <= 47)] = 40
    note_label[:, 3][(note_label[:, 3] >= 48) & (note_label[:, 3] <= 55)] = 48
    note_label[:, 3][(note_label[:, 3] >= 56) & (note_label[:, 3] <= 63)] = 56
    note_label[:, 3][(note_label[:, 3] >= 64) & (note_label[:, 3] <= 71)] = 64
    note_label[:, 3][(note_label[:, 3] >= 72) & (note_label[:, 3] <= 79)] = 72
    note_label[:, 3][(note_label[:, 3] >= 80) & (note_label[:, 3] <= 87)] = 80
    note_label[:, 3][(note_label[:, 3] >= 88) & (note_label[:, 3] <= 95)] = 88
    frame_label = np.zeros((len(frame_label_org), 128, len(instlist)))
    t = np.arange(0, len(frame_label) * H - H / 2, H)
    for k in range(len(instlist)):
        if np.any(note_label[:, 3] == instlist[k]):
            note_label_inst = note_label[note_label[:, 3] == instlist[k]]
            p_label = np.asarray(note_label_inst[:, 2], dtype=int)
            for i in range(len(note_label_inst)):
                onset = np.where(t >= note_label_inst[i, 0])[0][0]
                offset = np.where(t <= note_label_inst[i, 1])[0][-1]
                if onset > offset:
                    frame_label[onset, p_label[i], k] = 1
                else:
                    frame_label[onset:offset + 1, p_label[i], k] = 1
    return frame_label, note_label


def label_instfamily(output_path, label_file, instlist, H):
    with h5py.File(label_file, 'r') as label_f:
        onset_label_org = label_f['onset_label'][...]
        frame_label_org = label_f['frame_label'][...]
        note_label = label_f['note_label'][...]
    note_label[:, 3][(note_label[:, 3] >= 0) & (note_label[:, 3] <= 7)] = 0
    note_label[:, 3][(note_label[:, 3] >= 8) & (note_label[:, 3] <= 15)] = 8
    note_label[:, 3][(note_label[:, 3] >= 16) & (note_label[:, 3] <= 23)] = 16
    note_label[:, 3][(note_label[:, 3] >= 24) & (note_label[:, 3] <= 31)] = 24
    note_label[:, 3][(note_label[:, 3] >= 32) & (note_label[:, 3] <= 39)] = 32
    note_label[:, 3][(note_label[:, 3] >= 40) & (note_label[:, 3] <= 47)] = 40
    note_label[:, 3][(note_label[:, 3] >= 48) & (note_label[:, 3] <= 55)] = 48
    note_label[:, 3][(note_label[:, 3] >= 56) & (note_label[:, 3] <= 63)] = 56
    note_label[:, 3][(note_label[:, 3] >= 64) & (note_label[:, 3] <= 71)] = 64
    note_label[:, 3][(note_label[:, 3] >= 72) & (note_label[:, 3] <= 79)] = 72
    note_label[:, 3][(note_label[:, 3] >= 80) & (note_label[:, 3] <= 87)] = 80
    note_label[:, 3][(note_label[:, 3] >= 88) & (note_label[:, 3] <= 95)] = 88
    onset_label = np.zeros((len(onset_label_org), 128, len(instlist) + 1))
    frame_label = np.zeros((len(frame_label_org), 128, len(instlist) + 1))
    t = np.arange(0, len(onset_label) * H - H / 2, H)
    for k in range(len(instlist)):
        if np.any(note_label[:, 3] == instlist[k]):
            note_label_inst = note_label[note_label[:, 3] == instlist[k]]
            p_label = np.asarray(note_label_inst[:, 2], dtype=int)
            for i in range(len(note_label_inst)):
                onset = np.where(t >= note_label_inst[i, 0])[0][0]
                offset = np.where(t <= note_label_inst[i, 1])[0][-1]
                onset_label[onset, p_label[i], k] = 1
                if onset > offset:
                    frame_label[onset, p_label[i], k] = 1
                else:
                    frame_label[onset:offset + 1, p_label[i], k] = 1
    onset_label[:, :, -1] = np.where(np.max(onset_label[:, :, :-1], 2) > 0, 0, 1)
    frame_label[:, :, -1] = np.where(np.max(frame_label[:, :, :-1], 2) > 0, 0, 1)
    with h5py.File(output_path + os.path.basename(label_file), 'w') as f:
        f.create_dataset('onset_label', data=onset_label, chunks=(128, 128, np.shape(onset_label)[-1]),
                         compression='gzip', shuffle=True, fletcher32=True)
        f.create_dataset('frame_label', data=frame_label, chunks=(128, 128, np.shape(frame_label)[-1]),
                         compression='gzip', shuffle=True, fletcher32=True)
        f.create_dataset('note_label', data=note_label, compression='gzip', shuffle=True, fletcher32=True)


def parallel_label_instfamily(filename, instlist, H):
    output_path = os.path.dirname(os.path.dirname(filename)) + '/label-instfamily/'
    os.makedirs(output_path, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        for label_file in glob.glob(filename):
            executor.submit(label_instfamily, output_path, label_file, instlist, H)


def framelevel_evaluate(frame, frame_label, instlist, percussion_on=False):
    feval = np.zeros(3, dtype=np.int64)
    if not percussion_on:
        instlist = np.array(instlist)
        if np.any(instlist == 128):
            frame = frame[:, :, :-1]
            frame_label = frame_label[:, :, :-1]
    if frame.ndim == 3:
        frame_sum = np.sum(frame, 2)
        frame_sum[frame_sum > 1] = 1
    else:
        frame_sum = frame
    frame_label_sum = np.sum(frame_label, 2)
    frame_label_sum[frame_label_sum > 1] = 1
    feval[0] = np.sum(frame_sum * frame_label_sum)  # TP
    feval[1] = np.sum(frame_sum) - feval[0]  # FP
    feval[2] = np.sum(frame_label_sum) - feval[0]  # FN
    return feval


def notelevel_evaluate(note, note_label, accrange=accrange, percussive_on=False):
    neval = np.zeros(3, dtype=np.int64)
    if not percussive_on:
        note_label = note_label[note_label[:, 3] != 128]
    for i in range(0, 128):
        tref = note_label[:, 0][note_label[:, 2] == i]
        test = note[:, 0][note[:, 2] == i]
        for j in range(len(tref)):
            if any(np.abs(tref[j] - test) <= accrange):
                neval[0] += 1  # TP
                test[np.where(np.abs(tref[j] - test) <= accrange)[0][0]] = -1
    neval[1] = len(note[:, 0]) - neval[0]  # FP
    neval[2] = len(note_label[:, 0]) - neval[0]  # FN
    return neval


def framestream_evaluate(framestream, framestream_label, instlist, percussive_on=False):
    feval = np.zeros(3, dtype=np.int64)
    if not percussive_on:
        instlist = np.array(instlist)
        if np.any(instlist == 128):
            framestream = framestream[:, :, :-1]
            framestream_label = framestream_label[:, :, :-1]
    feval[0] = np.sum(framestream * framestream_label)  # TP
    feval[1] = np.sum(framestream) - feval[0]  # FP
    feval[2] = np.sum(framestream_label) - feval[0]  # FN
    return feval


def notestream_evaluate(notestream, note_label, instlist, accrange=accrange, percussive_on=False):
    neval = np.zeros(3, dtype=np.int64)
    if not percussive_on:
        note_label = note_label[note_label[:, 3] != 128]
    for k in instlist:
        note_inst = notestream[notestream[:, 3] == k]
        note_label_inst = note_label[note_label[:, 3] == k]
        for i in range(0, 128):
            tref = note_label_inst[:, 0][note_label_inst[:, 2] == i]
            test = note_inst[:, 0][note_inst[:, 2] == i]
            for j in range(len(tref)):
                if any(np.abs(tref[j] - test) <= accrange):
                    neval[0] += 1  # TP
                    test[np.where(np.abs(tref[j] - test) <= accrange)[0][0]] = -1
    neval[1] = len(notestream[:, 0]) - neval[0]  # FP
    neval[2] = len(note_label[:, 0]) - neval[0]  # FN
    return neval


def framelevel_evaluate_instwise(framestream, framestream_label):
    feval_instwise = np.zeros((np.shape(framestream_label)[2], 3), dtype=np.int64)
    for k in range(np.shape(framestream_label)[2]):
        if np.sum(framestream[:, :, k]) + np.sum(framestream_label[:, :, k]) == 0:
            continue
        feval_instwise[k][0] = np.sum(framestream[:, :, k] * framestream_label[:, :, k])  # TP
        feval_instwise[k][1] = np.sum(framestream[:, :, k]) - feval_instwise[k][0]  # FP
        feval_instwise[k][2] = np.sum(framestream_label[:, :, k]) - feval_instwise[k][0]  # FN
    return feval_instwise


def notelevel_evaluate_instwise(notestream, note_label, instlist, accrange=accrange):
    neval_instwise = np.zeros((len(instlist), 3), dtype=np.int64)
    for k in range(len(instlist)):
        if np.all(note_label[:, 3] != instlist[k]):
            if np.any(notestream[:, 3] == instlist[k]):
                note_inst = notestream[notestream[:, 3] == instlist[k]]
                neval_instwise[k][1] = len(note_inst[:, 0])
            continue
        note_label_inst = note_label[note_label[:, 3] == instlist[k]]
        if np.all(notestream[:, 3] != instlist[k]):
            neval_instwise[k][2] = len(note_label_inst[:, 0])
            continue
        note_inst = notestream[notestream[:, 3] == instlist[k]]
        for i in range(0, 128):
            tref = note_label_inst[:, 0][note_label_inst[:, 2] == i]
            test = note_inst[:, 0][note_inst[:, 2] == i]
            for j in range(len(tref)):
                if any(np.abs(tref[j] - test) <= accrange):
                    neval_instwise[k][0] += 1  # TP
                    test[np.where(np.abs(tref[j] - test) <= accrange)[0][0]] = -1
        neval_instwise[k][1] = len(note_inst[:, 0]) - neval_instwise[k][0]  # FP
        neval_instwise[k][2] = len(note_label_inst[:, 0]) - neval_instwise[k][0]  # FN
    return neval_instwise


def framelevel_evaluate_instwise_tpfn(frame, frame_label):
    feval_instwise = np.zeros((np.shape(frame_label)[2], 2), dtype=np.int64)
    for k in range(np.shape(frame_label)[2]):
        feval_instwise[k][0] = np.sum(frame * frame_label[:, :, k])  # TP
        feval_instwise[k][1] = np.sum(frame_label[:, :, k]) - feval_instwise[k][0]  # FN
    return feval_instwise


def midi_file_create(notestream, file_name):
    notestream[:, 1] += 1e-7
    onset_midi = np.append(np.delete(notestream, 1, 1), np.zeros((1, len(notestream))).reshape(-1, 1), axis=1)
    offset_midi = np.append(np.delete(notestream, 0, 1), np.ones((1, len(notestream))).reshape(-1, 1), axis=1)
    note_midi = np.append(onset_midi, offset_midi, axis=0)
    note_midi = note_midi[np.argsort(note_midi[:, 0])]

    bpm = 120
    ticks = 200  # 60 / (bpm * H)
    program_number = np.unique(note_midi[:, 2])
    mid = MidiFile(ticks_per_beat=ticks)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    track.append(MetaMessage('end_of_track', time=0))
    for p in range(len(program_number)):
        note_midi_inst = note_midi[note_midi[:, 2] == program_number[p]]
        track = MidiTrack()
        mid.tracks.append(track)
        if np.round(program_number[p]) == 128:
            chan = 9
            track.append(Message('program_change', channel=chan, program=0, time=0))
        elif p < 9:
            chan = p
            track.append(Message('program_change', channel=chan, program=np.round(program_number[p]).astype(int),
                                 time=0))
        else:
            chan = p + 1
            track.append(Message('program_change', channel=chan, program=np.round(program_number[p]).astype(int),
                                 time=0))
        for i in range(len(note_midi_inst)):
            if note_midi_inst[i, 3] == 0:
                if i == 0:
                    track.append(Message('note_on', channel=chan, note=np.round(note_midi_inst[i, 1]).astype(int),
                                         time=round(mido.second2tick(note_midi_inst[i, 0], ticks, mido.bpm2tempo(bpm))),
                                         velocity=127))
                else:
                    track.append(Message('note_on', channel=chan, note=np.round(note_midi_inst[i, 1]).astype(int),
                                         time=round(mido.second2tick(note_midi_inst[i, 0] - note_midi_inst[i - 1, 0],
                                                                     ticks, mido.bpm2tempo(bpm))), velocity=127))
            else:
                track.append(Message('note_off', channel=chan, note=np.round(note_midi_inst[i, 1]).astype(int),
                                     time=round(mido.second2tick(note_midi_inst[i, 0] - note_midi_inst[i - 1, 0], ticks,
                                                                 mido.bpm2tempo(bpm))), velocity=0))
        track.append(MetaMessage('end_of_track', time=0))
    mid.save(file_name + '.mid')
