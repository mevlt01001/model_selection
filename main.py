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
import os



import torch
import torch.nn as nn

class CustomClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomClassifier, self).__init__()
        self.first_layer = 200
        self.second_layer = 100
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if not hasattr(self, 'fc1'):
            _, input_features = x.view(x.size(0), -1).shape
            self.fc1 = nn.Linear(input_features, self.first_layer)
            self.fc2 = nn.Linear(self.first_layer, self.second_layer)
            self.fc3 = nn.Linear(self.second_layer, self.num_classes)
            self.to(x.device)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    

models = get_model()
data_dir = get_data_dir()
test_size, val_size = test_val_size()
dataset = create_dataset(data_dir, test_size=test_size, val_size=val_size)

loss_fn = nn.CrossEntropyLoss()
num_epochs = 10


for model_name, model, input_size, features in models:    
    # device free
    torch.cuda.empty_cache()
    clear()
    print(f"{'Model':.<20}: {model_name}")
    print(f"{'Input Size':.<20}: {input_size}")
    print(f"{'Features':.<20}: {features}")
    #GPU info
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

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.classifier = CustomClassifier(num_classes)
    model.to(device)    

    train_and_evaluate(model, model_name, dataloaders, loss_fn, optimizer, num_epochs, device)