import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_data_loaders(data_dir, input_size, batch_size=32, num_workers=4):

    # Dönüşümler
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Veri kümeleri
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms['val'])
    }

    # DataLoader'lar
    data_loaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    num_classes = len(image_datasets['train'].classes)

    return data_loaders, num_classes