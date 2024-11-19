# -*- coding: utf-8 -*-
"""
python3 classif_reports.py baseline chr2024/cnn
"""

import json
import os
from sklearn.metrics import classification_report
import numpy as np

MODEL_PP = {"resnet18": "ResNet18",
            "resnet50": "ResNet50",
            "densenet121": "DenseNet121",
            "mobilenet_v2": "MobileNetV2",
            "blip2" : "BLIP-2 & ViT-L & 14",
            "google_siglip-base-patch16-224" : "SigLIP & ViT-B & 16",
            "google_siglip-large-patch16-256" : " & ViT-L & 16",
            "google_siglip-so400m-patch14-384" : " & SO-400 & 14",
            "openai_clip-vit-base-patch32" : "CLIP & ViT-B & 32",
            "openai_clip-vit-base-patch16" : " & ViT-B & 16",
            "openai_clip-vit-large-patch14" : " & ViT-L & 14",
            "siglip" : "siglip-debug"
            }

def classif_report(true_classes, logits, target_names):
    preds = np.argmax(np.array(logits), axis=1)
    return classification_report(true_classes, preds, target_names=target_names,
                                output_dict=True)

CNN_MODELS = ["resnet18", "resnet50", "densenet121", "mobilenet_v2"]
CNN_SET2 = "aerial_ground_raised_predictions.json"
CNN_SET1 = "exterior_interior_predictions.json"
TF_SET2 = "gro_rai_aer"
TF_SET1 = "int_ext"

def collect_cnn_reports(base_dir):
    cnn_f1 = dict((model, {}) for model in CNN_MODELS)
    for model in CNN_MODELS:
        print(model)
        for catset in [CNN_SET1, CNN_SET2]:
            for split in ["0", "1", "2", "3", "4"]:
                fn = os.path.join(base_dir, model + "_reports", split, catset)
                with open(fn) as f:
                    report = json.load(f)
                rep_dict = classif_report(report["true_class"], report["probabilities"],
                               report["class_names"])
                for cl in report["class_names"]:
                    if cl not in cnn_f1[model]:
                        cnn_f1[model][cl] = 0.0
                    cnn_f1[model][cl] += rep_dict[cl]["f1-score"]
    return cnn_f1

def extract_ps(probs, ps, n_cl):
    return [row[ps * n_cl:(ps + 1) * n_cl] for row in probs]

def collect_mm_reports(base_dir):
    tf_f1 = {}
    for n_cl, catset in [(2, TF_SET1), (3, TF_SET2)]:
        for split in ["0", "1", "2", "3", "4"]:
            split_path = os.path.join(base_dir, catset + split)
            for fn in os.listdir(split_path):
                model = "_".join(fn.split("_")[:-1])
                if model not in tf_f1:
                    tf_f1[model] = {}
                with open(os.path.join(split_path, fn)) as f:
                    report = json.load(f)
                n_ps = int(len(report["prompts"]) / n_cl)
                for ps in range(n_ps):
                    probs = extract_ps(report["logits"], ps, n_cl)
                    rep_dict = classif_report(report["true_class"], probs,
                                   report["class_names"])
                    if ps not in tf_f1[model]:
                        tf_f1[model][ps] = {}
                    for cl in report["class_names"]:
                        if cl not in tf_f1[model][ps]:
                            tf_f1[model][ps][cl] = 0.0
                        tf_f1[model][ps][cl] += rep_dict[cl]["f1-score"]
    return tf_f1

#SEPA = "\t"
#ENDL = ""
SEPA = " & "
ENDL = " \\\\"
def baseline_table(cnn_f1):
    cats = ["interior", "exterior", "ground", "raised", "aerial"]
    print(SEPA.join([" "] + cats))
    for model in CNN_MODELS:
        f1s = ["{:.2f}".format(cnn_f1[model][cl] / 5.0) for cl in cats]
        print(SEPA.join(["    " + MODEL_PP[model]] + f1s) + ENDL)

def mm_summary_table(tf_f1, cat_set):
    print(SEPA.join([" ", " ", " "] + [str(ps) for ps in range(7)]))
    for model in tf_f1:
        f1s = []
        for ps in range(7):
            cl_f1 = sum(tf_f1[model][ps][cl] / 5.0 for cl in cat_set) / len(cat_set)
            f1s.append("{:.2f}".format(cl_f1))
        print(SEPA.join(["    " + MODEL_PP[model]] + f1s) + ENDL)

def usage(prog):
    print("usage: {} baseline|mm <path to reports>".format(prog))
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        if sys.argv[1] == "baseline":
            baseline_table(collect_cnn_reports(sys.argv[2]))
        elif sys.argv[1] == "mm":
            all_f1 = collect_mm_reports(sys.argv[2])
            mm_summary_table(all_f1, ["interior", "exterior"])
            mm_summary_table(all_f1, ["ground", "raised", "aerial"])
        else:
            usage(sys.argv[0])
    else:
        usage(sys.argv[0])
