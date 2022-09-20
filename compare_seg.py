import numpy as np
import os
 
def select(fnames):

    dices = np.zeros((len(fnames), 57, 3))
    for i, fn in enumerate(fnames):
        with open(fn, 'r') as f:
            contents = f.readlines()
        for j, c in enumerate(contents):
            newc = c.split(' ')
            for k, nc in enumerate(newc):
                dices[i, j, k] = float(nc)
    # print(np.mean(dices, axis=1))
    print(dices.shape)
    bests = {}
    names = []
    with open("/scratch/wenhuicu/TransBTS/data/test_list.txt", 'r') as f:
        for line in f:
            line = line.strip()
            name = line.split('/')[-1]
            names.append(name)

    for i in range(dices.shape[1]):
        be = dices[2, i]
        gce = dices[3, i]
        low = dices[0, i]
        ce = dices[1, i]
        diff_be = [be - low, be - ce]
        diff_gce = [gce - low, gce - ce]
        # print(diff_gce)
        if be[0] > 0.7 and np.min(be) > 0.5 and np.max(diff_be[0]) > 0.1 and np.max(diff_be[1]) > 0.05:
            bests[names[i]] = diff_be
            print(names[i], i, be)
        if gce[0] > 0.7 and np.min(be) > 0.5 and np.max(diff_gce[0]) > 0.1 and np.max(diff_gce[1]) > 0.05:
            bests[names[i]] = diff_gce
    print(bests)

select(['train0.3new.txt', 'cr0.7be0.txt', 'cr0.7be0.001.txt', 'cr0.7gce0.7.txt'])