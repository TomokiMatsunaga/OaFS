import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import argparse

H = 0.02  # hop size [s]
evaluation_on = 1  # 0 : off   1 : on
pianoroll_type = 1  # 0 : frame  1 : note


def plt_pianoroll(pr, H=H):
    pact = np.where(pr == 1)
    tact = pact[0] * H

    fig = plt.figure()
    plt.scatter(tact, pact[1], color='k', s=5)
    plt.xlim(0, tact[-1])
    plt.ylim(21, 108)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.ylabel('Piano roll', fontsize=18)
    plt.show()
    return fig


def plt_pianoroll_comparison(pr, pr_label, H=H):
    pestact = np.where(pr == 1)
    t_pestact = pestact[0] * H

    prefact = np.where(pr_label == 1)
    t_prefact = prefact[0] * H

    fig1 = plt.figure()
    plt.scatter(t_pestact, pestact[1], color='k', s=5)
    plt.xlim(0, max(t_pestact[-1], t_prefact[-1]))
    plt.ylim(21, 108)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.ylabel('Piano roll', fontsize=18)
    plt.title('prediction', fontsize=18)
    plt.show()
    fig2 = plt.figure()
    plt.scatter(t_prefact, prefact[1], color='k', s=5)
    plt.xlim(0, max(t_pestact[-1], t_prefact[-1]))
    plt.ylim(21, 108)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.ylabel('Piano roll', fontsize=18)
    plt.title('ground truth', fontsize=18)
    plt.show()
    return fig1, fig2


def plt_pianoroll_note(notestream):
    instnum = np.unique(notestream[:, 3]).astype(int)
    instnum = instnum[instnum != 128]
    cmap = plt.get_cmap("tab20")

    fig = plt.figure()
    ax = plt.axes()
    for j in range(len(instnum)):
        note_inst = notestream[notestream[:, 3] == instnum[j]]
        for i in range(len(note_inst)):
            # note = np.arange(note_inst[i, 0], note_inst[i, 1], H)
            # pitch = np.tile(note_inst[i, 2], len(note))
            # plt.plot(note, pitch, color=cmap(j))
            rect = patches.Rectangle((note_inst[i, 0], note_inst[i, 2] - 0.5), note_inst[i, 1] - note_inst[i, 0],
                                     1, linewidth=1, edgecolor='k', facecolor=cmap(j))
            ax.add_patch(rect)
    # plt.xlim(0, np.max(notestream[:, 1]))
    plt.xlim(0, 30)
    plt.ylim(21, 108)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.ylabel('Piano roll', fontsize=18)
    plt.show()
    return fig


def plt_pianoroll_note_comparison(notestream, note_label):
    instnum = np.unique(notestream[:, 3]).astype(int)
    instnum = instnum[instnum != 128]
    instnum_label = np.unique(note_label[:, 3]).astype(int)
    instnum_label = instnum_label[instnum_label != 128]
    inst = np.union1d(instnum, instnum_label)
    cmap = plt.get_cmap("tab20")

    fig1 = plt.figure()
    ax1 = plt.axes()
    for j in range(len(inst)):
        note_inst = notestream[notestream[:, 3] == inst[j]]
        for i in range(len(note_inst)):
            # note = np.arange(note_inst[i, 0], note_inst[i, 1], H)
            # pitch = np.tile(note_inst[i, 2], len(note))
            # plt.plot(note, pitch, color=cmap(j))
            rect = patches.Rectangle((note_inst[i, 0], note_inst[i, 2] - 0.5), note_inst[i, 1] - note_inst[i, 0],
                                     1, linewidth=1, edgecolor='k', facecolor=cmap(j))
            ax1.add_patch(rect)
    # plt.xlim(0, max(np.max(notestream[:, 1]), np.max(note_label[:, 1])))
    plt.xlim(0, 30)
    plt.ylim(21, 108)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.ylabel('Piano roll', fontsize=18)
    plt.title('prediction', fontsize=18)
    plt.show()

    fig2 = plt.figure()
    ax2 = plt.axes()
    for j in range(len(inst)):
        note_inst = note_label[note_label[:, 3] == inst[j]]
        for i in range(len(note_inst)):
            # note = np.arange(note_inst[i, 0], note_inst[i, 1], H)
            # pitch = np.tile(note_inst[i, 2], len(note))
            # plt.plot(note, pitch, color=cmap(j))
            rect = patches.Rectangle((note_inst[i, 0], note_inst[i, 2] - 0.5), note_inst[i, 1] - note_inst[i, 0],
                                     1, linewidth=1, edgecolor='k', facecolor=cmap(j))
            ax2.add_patch(rect)
    # plt.xlim(0, max(np.max(notestream[:, 1]), np.max(note_label[:, 1])))
    plt.xlim(0, 30)
    plt.ylim(21, 108)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.ylabel('Piano roll', fontsize=18)
    plt.title('ground truth', fontsize=18)
    plt.show()
    return fig1, fig2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_on', type=int)
    parser.add_argument('--pianoroll_type', type=int)
    args = parser.parse_args()
    if args.evaluation_on is not None:
        if args.evaluation_on != 0 and args.evaluation_on != 1:
            raise Exception('Select 0 or 1 for evaluation_on')
        else:
            evaluation_on = args.evaluation_on
    if args.pianoroll_type is not None:
        if args.pianoroll_type != 0 and args.pianoroll_type != 1:
            raise Exception('Select 0 or 1 for pianoroll_type')
        else:
            pianoroll_type = args.pianoroll_type
    pr_kw = np.load('temp/pianoroll.npz')
    if pianoroll_type == 0:
        if evaluation_on == 0:
            fig = plt_pianoroll(pr_kw['prest'])
            fig.savefig('temp/pianoroll_est.png')
        elif evaluation_on == 1:
            pr_label_sum = np.sum(pr_kw['prref'], 2)
            pr_label_sum[pr_label_sum > 1] = 1
            fig1, fig2 = plt_pianoroll_comparison(pr_kw['prest'], pr_label_sum)
            fig1.savefig('temp/pianoroll_est.png')
            fig2.savefig('temp/pianoroll_ref.png')
    elif pianoroll_type == 1:
        if evaluation_on == 0:
            fig = plt_pianoroll_note(pr_kw['note_est'])
            fig.savefig('temp/pianoroll_ref.png')
        elif evaluation_on == 1:
            fig1, fig2 = plt_pianoroll_note_comparison(pr_kw['note_est'], pr_kw['note_ref'])
            fig1.savefig('temp/pianoroll_est.png')
            fig2.savefig('temp/pianoroll_ref.png')
