import torch
import torch.nn as nn
import torch.optim as optim
from src.create_dataset import create_dataset
from src.get_data_dir import get_data_dir
from src.get_models import get_model
from src.clear import clear
from src.create_dataLoader import create_data_loaders
from src.train import train_and_evaluate
from src.get_test_val_size import test_val_size
from src.log import log
import os



import torch
import torch.nn as nn

class CustomClassifier(nn.Module):
    def __init__(self, num_classes, in_size):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def classifier_info(self):
        info = f"fc1({self.fc1.in_features}, {self.fc1.out_features})\n"
        info += f"relu()\n"
        info += f"dropout(0.5)\n"
        info += f"fc2({self.fc2.in_features}, {self.fc2.out_features})\n"
        info += f"relu()\n"
        info += f"fc3({self.fc3.in_features}, {self.fc3.out_features})"
        return info


models = get_model()
data_dir = get_data_dir()
test_size, val_size = test_val_size()
dataset = create_dataset(data_dir, test_size=test_size, val_size=val_size)

loss_fn = nn.CrossEntropyLoss()
num_epochs = 10


for model_name, model, input_size, fc, in_size in models:    

    torch.cuda.empty_cache()
    clear()
    print(f"{'Model':.<20}: {model_name}")
    print(f"{'Input Size':.<20}: {input_size}")

    
    if torch.cuda.is_available():
        print(f"{'GPU name':.<20}: {torch.cuda.get_device_name(0)}")
        print(f"{'GPU memory':.<20}: {(torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024)):0.3f} GB")
    else:
        print("GPU is not available")

    dataloaders, num_classes = create_data_loaders(dataset, input_size=input_size, batch_size=16)
    train_class_counts = {cls: 0 for cls in dataloaders['train'].dataset.classes}

    for _, target in dataloaders['train'].dataset.samples:
        class_name = dataloaders['train'].dataset.classes[target]
        train_class_counts[class_name] += 1

    val_class_counts = {cls: 0 for cls in dataloaders['val'].dataset.classes}

    for _, target in dataloaders['val'].dataset.samples:
        class_name = dataloaders['val'].dataset.classes[target]
        val_class_counts[class_name] += 1

    test_class_counts = {cls: 0 for cls in dataloaders['test'].dataset.classes}

    for _, target in dataloaders['test'].dataset.samples:
        class_name = dataloaders['test'].dataset.classes[target]
        test_class_counts[class_name] += 1

    print(f"{'CLASSES':*^50} ")    
    for class_name in train_class_counts:
        train_count = train_class_counts[class_name]
        val_count = val_class_counts[class_name]
        test_count = test_class_counts[class_name]
        print(f"{class_name:<20} train {train_count}, val {val_count}, test {test_count}")
    print("\n")

    print(f"{'Model':.<20}: {model_name}")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"{'Optimizer':.<20}: {optimizer}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'Device':.<20}: {device}")

    if fc:
        model.fc = CustomClassifier(num_classes=num_classes, in_size=in_size)
        print(f"{'Classifier':.<20}: {model.fc.classifier_info()}")
    else:
        model.classifier = CustomClassifier(num_classes=num_classes, in_size=in_size)
        print(f"{'Classifier':.<20}: {model.classifier.classifier_info()}")
    
    model.to(device)    

    train_and_evaluate(model, model_name, dataloaders, loss_fn, optimizer, num_epochs, device)