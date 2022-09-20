import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

def plot_tool(ax, lowerbound, ce, bce, gce, sce, upperbound):
    # plt.figure(figsize=(10,6), tight_layout=True)

    # plt.title(title)
    x = [0.3, 0.5, 0.7]
    ax.plot(x, lowerbound, '-p')
    ax.plot(x, ce, '-p')
    ax.plot(x, bce, '-p')
    ax.plot(x, gce, '-p')
    ax.plot(x, sce, '-p')
    ax.plot(x, upperbound, '-p')
    

def plot_res():
    rates = ['0.3', '0.5', '0.7']
    models = ['baseline', 'ce', 'betace', 'gce', 'sce']
    upperbound = read_from_csv(path + 'upperbound_f0.csv')
    res_all = np.zeros((3, 5, 6))
    for i, r in enumerate(rates):
        for j, m in enumerate(models):
            mean, var = calc_mean_var(m+r)
            res_all[i, j, :] = mean
    
    classes = ['wt', 'tc', 'et']

    fig, ax = plt.subplots(1, 1)
    for j, c in enumerate(classes):
        # figname = c + '_dice.png'
        plot_tool(ax, res_all[:, 0, j], res_all[:, 1, j], res_all[:, 2, j], res_all[:, 3, j], res_all[:, 4, j], np.array([upperbound[j]]*3))
        
    plt.legend(["Lowerbound", "CE", "BCE", "GCE", "SCE", "Upperbound"], loc="lower right", prop={"size":7})
    plt.savefig("dices.png", bbox_inches='tight')
plot_res()
# print(calc_mean_var('betace00010.3'))