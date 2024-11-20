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

It is recommended to install the LAVIS package in a separate virtual
enviroment to avoid module version clashes.

```
pip install salesforce-lavis
```


## Classification

This part documents the classification experiment. Fully reproducing
the experiment requires the classification dataset, which
we cannot legally distribute due to copyright. The experiment
can still be repeated on a similar dataset.

Preparation of cross-validation splits, assuming that the images in
the `ground_truth` directory are sorted into subdirectories named
after the classes (`interior` etc):

```
python3 scripts/splits.py path/to/ground_truth/ ${HOME}/chr2024/images/split_data/int_ext interior exterior
python3 scripts/splits.py path/to/ground_truth/ ${HOME}/chr2024/images/split_data/gro_rai_aer ground raised aerial
```

Generate supervised baseline results (the script won't work as-is,
edit the paths etc as necessary):

```
scripts/train_cnn.sh
```

Create a per-class summary table. This assumes report files for all
splits, classes and models have been generated.

```
python3 scripts/classif_reports.py baseline chr2024/cnn
```

Generate results with CLIP and SigLIP models (edit the script as necessary):

```
scripts/test_clip.sh
```

Generate results with the BLIP-2 model (separate venv assumed):

```
scripts/test_blip.sh
```

Create summary tables for all promptsets (in `prompts_db.json`) for
scene and viewpoint elevation.

```
python3 scripts/classif_reports.py mm chr2024/cnn/multimodal/reports
```

## Search

Rest of the content coming in Dec 2024.
