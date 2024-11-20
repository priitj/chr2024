import os
import json

from utils.data_utils import classes_from_subfolder_names, open_images
from utils.device_config import configure_device

def run_all_splits(model_name, base_dir, prompts_path, report_dir, device="cuda"):
    if "openai/clip" in model_name:
        from clip import CLIPClassifier
        classifier = CLIPClassifier(model_name, device=device)
    elif "google/siglip" in model_name:
        from clip import SigLIPClassifier
        classifier = SigLIPClassifier(model_name, device=device)
    elif "blip2" in model_name:
        from multimodal.blip2 import Blip2Classifier
        classifier = Blip2Classifier(model_name, device=device)
    else:
        raise NotImplementedError("model name not known")

    with open(prompts_path) as f:
        prompts_db = json.load(f)

    for split_num in range(5):
        for exp_name in ["int_ext", "gro_rai_aer"]:
            dirname = "{}{}".format(exp_name, split_num)
            print("Split", dirname)
            data_path = os.path.join(base_dir, dirname)
            report_path = os.path.join(report_dir, dirname)
            run_one_split(data_path, prompts_db[exp_name],
                          classifier, report_path)

def run_one_split(data_path, prompt_dicts, classifier, report_path):
    """Predict class probabilities for provided images."""

    # TODO: add synthetic prompts using vectorizer/wordlist

    test_path = os.path.join(data_path, "test")
    classes = classes_from_subfolder_names(test_path)

    # prompts should be aligned with class order
    prompts = []
    for pd in prompt_dicts:
        prompts += [pd[c] for c in classes]

    print("Opening images...")
    images = open_images(classes, test_path)

    image_names, true_classes, predictions = classifier.predictions(images,
                                                                    classes, prompts)
    results = {
        "model": classifier.model_name,
        "class_names": classes,
        "prompts": prompts,
        "image_names": image_names,
        "true_class": true_classes,
        "logits": predictions
    }
    pred_fn = "{}_predictions.json".format(classifier.model_name.replace("/", "_"))
    with open(os.path.join(report_path, pred_fn), "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    import sys
    device = configure_device()
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    model_name = sys.argv[1]
    base_dir = sys.argv[2]
    prompts_file = sys.argv[3]
    reports_dir = sys.argv[4]
    run_all_splits(model_name, base_dir, prompts_file, reports_dir,
                   device=device)
