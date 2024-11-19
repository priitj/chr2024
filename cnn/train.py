import json
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models
from utils.device_config import configure_device
from utils.data_utils import data_loaders, datasets
from utils.model_utils import train_model, predict

def make_classifier(n_features, n_classes):
    # minimal
    #return nn.Linear(n_features, n_classes)
    # MobileNetV2-style
    return nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_features, n_classes),
        )

if __name__ == '__main__':
    import sys
    device = configure_device()
    model_name = sys.argv[1]
    report_path = sys.argv[2]
    datasets = datasets(sys.argv[3])
    loaders = data_loaders(datasets, batch_size=64, num_workers=0, device=device)
    class_names = datasets['train'].classes
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
    exp_name = "_".join(class_names)

    if model_name == "resnet18":
        base_model = models.resnet18(weights='IMAGENET1K_V1')
    elif model_name == "resnet50":
        base_model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_name == "densenet121":
        base_model = models.densenet121(weights='IMAGENET1K_V1')
    elif model_name == "mobilenet_v2":
        base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # freeze the parameters
    for param in base_model.parameters():
        param.requires_grad = False

    if model_name in ["resnet18", "resnet50"]:
        base_model.fc = make_classifier(base_model.fc.in_features, len(class_names))
        classifier = base_model.fc
    elif model_name == "densenet121":
        base_model.classifier = make_classifier(base_model.classifier.in_features, len(class_names))
        classifier = base_model.classifier
    elif model_name == "mobilenet_v2":
        base_model.classifier = make_classifier(base_model.last_channel, len(class_names))
        classifier = base_model.classifier

    base_model = base_model.to(device)
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(classifier.parameters())
    # Decay LR by a factor of 0.1 every 7 epochs
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scheduler = None
    # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = train_model(base_model, criterion, optimizer, scheduler, loaders, device, dataset_sizes)
    # save_path = 'states/{}_{}_model.pth'.format(exp_name, model_name)
    # torch.save(model.state_dict(), save_path)

    yhat, y, probs = predict(model, class_names, loaders["test"], device)
    results = {
        "model": model_name,
        "class_names": class_names,
        "predicted_class": yhat,
        "true_class": y,
        "probabilities": probs
    }
    with open("{}/{}_predictions.json".format(report_path,
            exp_name), "w") as f:
        json.dump(results, f)

