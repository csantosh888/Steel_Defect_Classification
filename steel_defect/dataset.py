"""
Dataset module for steel defect classification.

Provides functions to scan the class-directory dataset structure,
create train/val/test splits, and a PyTorch Dataset class for
loading and transforming images.
"""
from pathlib import Path

import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from steel_defect.utils import setup_logging, CLASS_NAMES, DATA_DIR, IMAGE_SIZE

logger = setup_logging(__name__)

# File extensions to include when scanning for images
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def build_file_list(data_dir: Path | str | None = None) -> list[tuple[str, int]]:
    
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Place your images in subdirectories: {', '.join(CLASS_NAMES)}"
        )

    # ┌──────────────────────────────────────────────┐
    # │  DATA-1: Write your code below               │
    # └──────────────────────────────────────────────┘
    file_list=[]
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        if class_dir.name not in CLASS_NAMES:
            continue

        label_idx = CLASS_NAMES.index(class_dir.name)

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                file_list.append((str(img_path), label_idx))

    return sorted(file_list)
    

def create_splits(
    file_list: list[tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    
    # ┌──────────────────────────────────────────────┐
    # │  DATA-3: Write your code below               │
    # └──────────────────────────────────────────────┘
    labels = [label for _, label in file_list]
    test_ratio = 1 - train_ratio - val_ratio
    train_val_portion, test_list = train_test_split(
        file_list,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed
    )
    train_val_labels = [label for _, label in train_val_portion]
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_list, val_list = train_test_split(
        train_val_portion,
        test_size=val_relative,
        stratify=train_val_labels,
        random_state=seed
    )

    return train_list, val_list, test_list

    

class SteelDataset(Dataset):
    """
    PyTorch Dataset for steel defect images.

    Loads images from disk, applies transforms, and returns
    (tensor, label) pairs ready for the DataLoader.
    """

    def __init__(
        self,
        file_list: list[tuple[str, int]],
        transform: A.Compose | None = None,
    ):
        """
        Args:
            file_list: List of (image_path, label) tuples.
            transform: Albumentations transform pipeline to apply.
        """
        self.file_list = file_list
        self.transform = transform

        logger.info(
            "SteelDataset created | samples=%d | transform=%s",
            len(self.file_list),
            "yes" if transform else "none",
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        
        # ┌──────────────────────────────────────────────┐
        # │  DATA-2: Write your code below               │
        # └──────────────────────────────────────────────┘
        image_path, label = self.file_list[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
    
       

# ── Scaffold — DataLoader helper ──────────────────────────────

def create_dataloaders(
    train_list: list,
    val_list: list,
    train_transform: A.Compose,
    val_transform: A.Compose,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    
    train_ds = SteelDataset(train_list, transform=train_transform)
    val_ds = SteelDataset(val_list, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders created | train=%d batches | val=%d batches | batch_size=%d",
        len(train_loader),
        len(val_loader),
        batch_size,
    )

    return train_loader, val_loader
