import torch
from torch.nn import Module
from sklearn.metrics import accuracy_score
from src.log import log
import time


def calculate_model_size(model: Module):
    """Modelin VRAM'da ne kadar yer kaplayacağını hesaplar."""
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()  
    return total_size / (1024 ** 2) 

def train_and_evaluate(model, model_name, dataloaders, loss_fn, optimizer, num_epochs, device, train_log_file="train_log.txt"):
    where = "FUNC_TRAIN_AND_EVALUATE"

    model_size_mb = calculate_model_size(model)
    available_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    
    if model_size_mb > available_vram * 0.8:  
        error_message = f"[{model_name.upper()} ERROR]: Model requires {model_size_mb:.2f} MB VRAM, but only {available_vram * 0.8:.2f} MB is available."
        log(where, error_message, train_log_file)
        print(error_message)
        return

    log(where, f"[{model_name.upper()} INFO]: Model size {model_size_mb:.2f} MB. Proceeding with training.", train_log_file)

    for epoch in range(num_epochs):
        try:
            model.train()
            total_loss = 0
            log(where, f"[INFO]: Starting epoch {epoch+1}/{num_epochs}...")

            for batch_idx, (images, labels) in enumerate(dataloaders['train']):
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                loss = loss_fn(output, labels)

                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step()  

                total_loss += loss.item()
                log(where, f"[{model_name.upper()} INFO]: Batch {batch_idx}: Loss = {loss.item():.4f}", train_log_file)

            avg_loss = total_loss / len(dataloaders['train'])
            log(where, f"[{model_name.upper()} INFO]: Epoch {epoch+1} completed. Average Loss = {avg_loss:.4f}", train_log_file)

            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in dataloaders['val']:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    _, preds = torch.max(output, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            log(where, f"[{model_name.upper()} INFO]: Epoch {epoch+1} Validation Accuracy = {accuracy:.4f}", train_log_file)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        except Exception as e:
            error_message = f"[{model_name.upper()} ERROR]: An error occurred during epoch {epoch+1}. {str(e)}"
            log(where, error_message, train_log_file)
            print(error_message)