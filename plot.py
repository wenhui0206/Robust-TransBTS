import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
path = '/scratch1/wenhuicu/robust_seg/TransBTS_outputs/csv_results/'

def read_from_csv(fname):
    df = pd.read_csv(fname)
    wt_dc = df["WT dice"][0]
    wt_hd = df["WT hd"][0]
    tc_dc = df["TC dice"][0]
    tc_hd = df["TC hd"][0]
    en_dc = df["ET dice"][0]
    en_hd = df["ET hd"][0]
    return [wt_dc, tc_dc, en_dc, wt_hd, tc_hd, en_hd]


def calc_mean_var(model_name):
    folds = ['_f0', '_f1', '_f2']
    res = []
    for f in folds:
        fn = path + model_name + f + '.csv'
        if os.path.exists(fn):
            res.append(read_from_csv(fn))
    res = np.array(res)
    mean_res = np.mean(res, axis=0)
    var_res = np.std(res, axis=0)

    return mean_res, var_res

def plot_tool(ax, lowerbound, ce, bce, gce, sce, upperbound, colors):
    # plt.figure(figsize=(10,6), tight_layout=True)

    # plt.title(title)
    x = [0.3, 0.5, 0.7]
    ax.plot(x, lowerbound, '-p', color=colors[0])
    ax.plot(x, ce, '-p', color=colors[1])
    ax.plot(x, bce, '-p', color=colors[2])
    ax.plot(x, gce, '-p', color=colors[3])
    ax.plot(x, sce, '-p', color=colors[4])
    ax.plot(x, upperbound, '-p', color=colors[5])


def plot_tool2(figname, lowerbound, ce, bce, gce, sce, upperbound):
    sns.set_style('darkgrid')
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)
    plt.figure(figsize=(8,6), tight_layout=True)
    # plt.figure()
    # plt.title(title)
    plt.xticks([0.3, 0.5, 0.7])
    gap = (upperbound[0] - lowerbound[0]) / 4
    # plt.locator_params(axis="y", nbins=4)
    # if(figname[:2] == "WT"):

    plt.yticks(np.arange(round(lowerbound[0], 2), round(lowerbound[0], 2)+0.07, 0.01))
    plt.xlabel('Value of p')
    plt.ylabel("Dice Score")
    colorblue = sns.color_palette("ch:s=.25,rot=-.25")
    colorgreen = sns.color_palette("light:#5A9")
    colored = sns.color_palette("flare")
    x = [0.3, 0.5, 0.7]
    lw = 4
    plt.plot(x, lowerbound, '-p', color=colorblue[1], linewidth=lw)
    plt.plot(x, ce, '-p', color=colored[0], linewidth=lw)
    plt.plot(x, bce, '-p', color=colorblue[3], linewidth=lw)
    plt.plot(x, gce, '-p', color=colorgreen[4], linewidth=lw)
    plt.plot(x, sce, '-p', color=colorblue[5], linewidth=lw)
    plt.plot(x, upperbound, '-p', color=colorblue[2], linewidth=3)
    plt.legend(["Lower bound", "CE", "BCE", "GCE", "SCE", "Upper bound"], loc="lower right")
    plt.savefig(figname, bbox_inches='tight', format='eps')


def plot_res():
    rates = ['0.3', '0.5', '0.7']
    models = ['baseline', 'ce', 'betace', 'gce', 'sce']
    upperbound = read_from_csv(path + 'upperbound_f0.csv')
    res_all = np.zeros((3, 5, 6))
    for i, r in enumerate(rates):
        for j, m in enumerate(models):
            if r == '0.3' and m == 'betace':
                m = 'betace0001'
            if r == '0.5' and m == 'betace':
                m = 'betace001'
            mean, var = calc_mean_var(m+r)
            res_all[i, j, :] = mean
    
    classes = ['WT', 'TC', 'ET']
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(3, hspace=0.05)
    axes = gs.subplots(sharex=True)
    plt.xticks([0.3, 0.5, 0.7])
    plt.xlabel('Percentage of Labeled Data')
    plt.ylabel("Mean Dice Score")
    colors = sns.color_palette('pastel')
    for j, c in enumerate(classes):
        # figname = c + '_dice.png'
        plot_tool(axes[j], res_all[:, 0, j], res_all[:, 1, j], res_all[:, 2, j], res_all[:, 3, j], res_all[:, 4, j], np.array([upperbound[j]]*3), colors)
        axes[j].label_outer()
    plt.legend(["Lowerbound", "CE", "BCE", "GCE", "SCE", "Upperbound"], loc="lower right", prop={"size":8}, labelspacing=0.2, handlelength=2)
    plt.savefig("dices.png", bbox_inches='tight')


def plot_res2():
    rates = ['0.3', '0.5', '0.7']
    models = ['baseline', 'ce', 'betace0001', 'gce', 'sce100']
    upperbound = read_from_csv(path + 'upperbound_f0.csv')
    res_all = np.zeros((3, 5, 6))
    for i, r in enumerate(rates):
        for j, m in enumerate(models):
            mean, var = calc_mean_var(m+r)
            res_all[i, j, :] = mean
    
    classes = ['WT', 'TC', 'ET']
    
    for j, c in enumerate(classes):
        figname = c + '_dice.eps'
        plot_tool2(figname, res_all[:, 0, j], res_all[:, 1, j], res_all[:, 2, j], res_all[:, 3, j], res_all[:, 4, j], np.array([upperbound[j]]*3))
        

# plot_res2()

mean, var = calc_mean_var('CElw07')
for m, v in zip(mean[:3], var[:3]):
    print("{:.4f}".format(m), "(", "{:.3f}".format(v), ")")