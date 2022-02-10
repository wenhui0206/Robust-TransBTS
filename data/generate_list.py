import os
import random
import pdb


def generate_train_all(root):
    train_dir1 = os.listdir(root + 'Training/HGG/')
    train_dir2 = os.listdir(root + 'Training/LGG/')
    with open('train_all_list.txt', 'w') as f:
        for fn in train_dir1:
            f.write(fn + '\n')
        for fn in train_dir2:
            f.write(fn + '\n')
    
    # test_dir = os.listdir(root + 'Testing/')

    # with open('test_list.txt', 'w') as f:
    #     for fn in test_dir:
    #         f.write(fn + '\n')

def split_train_val_te():
    with open('train_all_list.txt', 'r') as f:
        fnames = f.readlines()
    print(fnames)
    random.shuffle(fnames)
    pdb.set_trace()

    total_num = len(fnames)
    num_val = int(0.1 * total_num)
    num_test = int(0.2 * total_num)

    for i, fn in enumerate(fnames):
        if i < num_val:
            with open('val_list.txt', 'a+') as f:
                f.write(fn)
        if i >= num_val and i < num_val + num_test:
            with open('test_list.txt', 'a+') as f:
                f.write(fn)
        if i >= num_val + num_test:
            with open('train_list.txt', 'a+') as f:
                f.write(fn)
    

if __name__ == '__main__':
    split_train_val_te()

