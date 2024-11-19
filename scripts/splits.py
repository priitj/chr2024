# -*- coding: utf-8 -*-
"""
Make train/val/test splits

```
python3 splits.py path/to/ground_truth/ ${HOME}/chr2024/images/split_data/int_ext interior exterior
```
"""

import os
import os.path
import random

K = 5

def mkdirs(prefix, classes):
    for split in ["train", "val", "test"]:
        for cl in classes:
            path = os.path.join(prefix, split, cl)
            os.makedirs(path, exist_ok=True)

def class_images(srcdir, cl):
    path = os.path.join(srcdir, cl)
    return os.listdir(path)

def make_split(imglist, train_ratio=0.8, val_ratio=0.1):
    N = len(imglist)
    N_train = N * train_ratio
    N_val = N_train + N * val_ratio
    indices = list(range(N))
    random.shuffle(indices)
    train_set, val_set, test_set = [], [], []
    for i in range(N):
        if i < N_train:
            train_set.append(imglist[indices[i]])
        elif i < N_val:
            val_set.append(imglist[indices[i]])
        else:
            test_set.append(imglist[indices[i]])
    return train_set, val_set, test_set

def make_cv_split(imglist, k=K, val_ratio=0.05):
    N = len(imglist)
    N_test = N // k
    N_val = int(N * val_ratio)
    indices = list(range(N))
    random.shuffle(indices)

    splits = []
    for j in range(k):
        test_s = j * N_test
        test_e = test_s + N_test
        test_indices = list(indices[test_s:test_e])
        train_indices = list(indices[:test_s]) + list(indices[test_e:])
        random.shuffle(train_indices)

        train_set, val_set, test_set = [], [], []
        for i in range(len(train_indices)):
            if i < N_val:
                val_set.append(imglist[train_indices[i]])
            else:
                train_set.append(imglist[train_indices[i]])
        for i in range(len(test_indices)):
            test_set.append(imglist[test_indices[i]])

        splits.append([train_set, val_set, test_set])
    return splits

def make_links(srcdir, prefix, classes):
    abspath = os.path.abspath(srcdir)
    for cl in classes:
        splits = make_split(class_images(srcdir, cl))
        for i, split in enumerate(["train", "val", "test"]):
            for fn in splits[i]:
                os.symlink(os.path.join(abspath, cl, fn),
                           os.path.join(prefix, split, cl, fn))
            print(" ", cl, split, len(splits[i]), "links created")

def make_cv_links(srcdir, prefix, classes):
    abspath = os.path.abspath(srcdir)
    for cl in classes:
        cvsplits = make_cv_split(class_images(srcdir, cl))
        for j, splits in enumerate(cvsplits):
            for i, split in enumerate(["train", "val", "test"]):
                for fn in splits[i]:
                    os.symlink(os.path.join(abspath, cl, fn),
                               os.path.join("{}{}".format(prefix, j), split, cl, fn))
                print(" ", j, cl, split, len(splits[i]), "links created")

if __name__ == "__main__":
    import sys
    srcdir = sys.argv[1]
    prefix = sys.argv[2]
    classes = sys.argv[3:]
    print(srcdir, prefix, classes)
    for j in range(K):
        mkdirs("{}{}".format(prefix, j), classes)
    make_cv_links(srcdir, prefix, classes)
