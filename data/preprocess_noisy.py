import pickle
import os
import numpy as np
import nibabel as nib
import pdb

modalities = ('flair', 't1ce', 't1', 't2')
suffix = '_f2'
# train
train_set = {
        'root': '/scratch/wenhuicu/TransBTS_outputs/output/pretrain0.3' + suffix + '/Trans_train0.3' + suffix,
        'flist': '/scratch/wenhuicu/TransBTS/data/train_list.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'root': 'path to validation set',
        'flist': 'valid.txt',
        'has_label': False
        }

test_set = {
        'root': 'path to testing set',
        'flist': 'test.txt',
        'has_label': False
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = path + 'data_i16.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return


def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    sub_name = path.split('/')[-1]
    if has_label:
        label = np.array(nib_load(path + '.nii.gz'), dtype='uint8', order='C')
    # images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]
    # print(path)
    # pdb.set_trace()
    output_path = '/scratch/wenhuicu/brats_preprocessed/'
    folder_dir = output_path + sub_name + '/'
    if os.path.exists(folder_dir) == False:
        os.mkdir(folder_dir)

    output = folder_dir + sub_name + '_noisy_label_0.3' + suffix + '.pkl'
    # mask = images.sum(-1) > 0
    # for k in range(4):

    #     x = images[..., k]  #
    #     y = x[mask]

    #     # 0.8885
    #     x[mask] -= y.mean()
    #     x[mask] /= y.std()
    #     images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump(label, f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    # file_list = os.path.join(root, dset['flist'])
    subjects = open(dset['flist']).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, name) for sub, name in zip(subjects, names)]

    cnt = 0
    for path in paths:
        if os.path.exists(path + '.nii.gz'):
            process_f32b0(path, has_label)
            cnt += 1
    print(cnt)


if __name__ == '__main__':
    doit(train_set)
    # doit(valid_set)
    # doit(test_set)

