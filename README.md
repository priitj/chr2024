# chr2024

The scripts for the classification and search experiments in the
conference paper:

E. Maksimova, M.-A. Meimer, M. Piirsalu, P. Järv. *Viability of Zero-shot Classification and Search of Historical Photos* CHR2024, Aarhus, Denmark, December 4-6, 2024.

## Dependencies

```
pip install torch torchvision
pip install transformers
pip install scikit-learn
```

## Classification

This part documents the classification experiment. Fully reproducing
the experiment requires the classification dataset, which
we cannot legally distribute due to copyright. The experiment
can still be repeated on a similar dataset.

Preparation of cross-validation splits, assuming that the `ground_truth`
directory contains classes in the `interior` and `exterior` subdirectories:

```
python3 scripts/splits.py path/to/ground_truth/ ${HOME}/chr2024/images/split_data/int_ext interior exterior
```

Generate supervised baseline results (the script won't work as-is,
edit the paths etc as necessary):

```
scripts/train_cnn.sh
```

Create a summary table. This assumes report files for all splits, class
sets and models have been generated.

```
python3 scripts/classif_reports.py baseline chr2024/cnn
```

Rest of the content coming in Dec 2024.
