import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Data augmentation and normalization for training
# Just normalization for validation (and testing)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def datasets(dataset_path):
    return {
        'train': ImageFolder(dataset_path + "train", transform=data_transforms['train']),
        'val': ImageFolder(dataset_path + "val", transform=data_transforms['val']),
        'test': ImageFolder(dataset_path + "test", transform=data_transforms['val']),
    }


def data_loaders(datasets, batch_size, num_workers, device='cuda'):
    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            generator=torch.Generator(device=device)),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          generator=torch.Generator(device=device)),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           generator=torch.Generator(device=device))
    }


def classes_from_subfolder_names(root):
    return [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

def open_images(classes, data_path):
    """Open all images from a provided image path."""
    images = []

    for c in classes:
        folder_path = os.path.join(data_path, c)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path).convert("RGB")
            images.append([img.copy(), filename, c])
            img.close()

    return images
