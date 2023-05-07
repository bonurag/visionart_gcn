import torch

from typing import List
from torch.utils.data import DataLoader
from torch_geometric.datasets import MNISTSuperpixels

# data constants
BATCH_SIZE = 64
NUM_WORKERS = 32

def build_mnist_superpixels_dataset(train: bool) -> MNISTSuperpixels:
    return MNISTSuperpixels(
        root="mnist-superpixels-dataset",
        train=train,
    )

def build_collate_fn(device: str):
    def collate_fn(original_batch):
        batch_node_features: List[torch.Tensor] = []
        batch_edge_indices: List[torch.Tensor] = []
        classes: List[int] = []

        for data in original_batch:
            node_features = torch.cat((data.x, data.pos), dim=-1).to(device)
            edge_indices = data.edge_index.to(device)
            class_ = int(data.y)

            batch_node_features.append(node_features)
            batch_edge_indices.append(edge_indices)
            classes.append(class_)

        collated = {
            "batch_node_features": batch_node_features,
            "batch_edge_indices": batch_edge_indices,
            "classes": torch.LongTensor(classes).to(device),
        }

        return collated

    return collate_fn
    
def build_dataloader(
    dataset: MNISTSuperpixels,
    batch_size: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=build_collate_fn(device=device),
    )

    return loader


def build_train_val_dataloaders():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = build_mnist_superpixels_dataset(train=True)
    valid_dataset = build_mnist_superpixels_dataset(train=False)
    test_dataset = build_mnist_superpixels_dataset(train=False)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        device=device,
    )

    valid_loader = build_dataloader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )

    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )

    return train_loader, valid_loader, test_loader