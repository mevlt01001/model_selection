import os
import shutil
from sklearn.model_selection import train_test_split
from src.clear import clear
from src.log import log

def create_dataset(root_dir, val_size=0.2):
    """
    Creates a dataset structure with training and validation splits.

    Parameters:
    - root_dir (str): Path to the source dataset.
    - val_size (float): Fraction of the dataset to be used for validation.

    Returns:
    - str: Path to the adjusted dataset directory.
    """
    aroot_dir = f'adjusted_{root_dir}'
    where = "FUNC_CREATE_DATASET"

    if os.path.exists(aroot_dir):
        log(where, f"[INFO]: Directory '{os.getcwd()}/{aroot_dir}' already exists.")
        print(f"[INFO]: Directory '{os.getcwd()}/{aroot_dir}' already exists.")
    elif not os.path.exists(root_dir):
        log(where, f"[ERROR]: Source directory '{os.getcwd()}/{root_dir}' does not exist.")
        print(f"[ERROR]: Source directory '{os.getcwd()}/{root_dir}' does not exist.")
        return None

    log(where, "[INFO]: Adjusting dataset...")
    train_dir = os.path.join(aroot_dir, "train")
    val_dir = os.path.join(aroot_dir, "val")

    log(where, "[INFO]: Creating train and validation folders...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            log(where, f"[INFO]: Processing class '{class_name}'...")
            images = os.listdir(class_path)

            try:
                train_images, val_images = train_test_split(images, test_size=val_size, random_state=42)
            except ValueError as e:
                log(where, f"[ERROR]: Unable to split images for class '{class_name}': {str(e)}")
                continue

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            for img in train_images:
                try:
                    shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
                except Exception as e:
                    log(where, f"[ERROR]: Failed to copy image '{img}' to train directory: {str(e)}")

            log(where, f"[INFO]: {len(train_images)} images copied to train/{class_name}")

            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            for img in val_images:
                try:
                    shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
                except Exception as e:
                    log(where, f"[ERROR]: Failed to copy image '{img}' to validation directory: {str(e)}")

            log(where, f"[INFO]: {len(val_images)} images copied to val/{class_name}")

    info_dataset(aroot_dir)
    log(where, f"[SUCCESS]: Dataset created at '{aroot_dir}'.")
    return aroot_dir

def info_dataset(root_dir):
    """
    Logs and prints information about the dataset structure.

    Parameters:
    - root_dir (str): Path to the dataset directory.
    """
    where = "FUNC_INFO_DATASET"
    for type in os.listdir(root_dir):
        log(where, f"[INFO]: {type} directory content:")
        print(f"{type:>7}:")
        for class_name in os.listdir(os.path.join(root_dir, type)):
            try:
                count = len(os.listdir(os.path.join(root_dir, type, class_name)))
                log(where, f"[INFO]: Class '{class_name}' contains {count} images.")
                print(f"\t{class_name:<9}: {count} images")
            except Exception as e:
                log(where, f"[ERROR]: Unable to count images in class '{class_name}': {str(e)}")