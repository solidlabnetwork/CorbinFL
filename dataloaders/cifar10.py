import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from .base import BaseDataLoader, DatasetSubset

class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, n_clients: int, batch_size: int, iid: bool = True):
        super().__init__(data_dir, n_clients, batch_size, iid)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def distribute_data(self, dataset):
        """Distribute data among clients (IID)"""
        num_items = int(len(dataset) / self.n_clients)
        all_idxs = np.arange(len(dataset))
        np.random.shuffle(all_idxs)
        return {i: set(all_idxs[i * num_items:(i + 1) * num_items]) 
                for i in range(self.n_clients)}

    def load_data(self):
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform
        )

        # Split train into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Distribute data among clients
        dict_users = self.distribute_data(train_dataset)

        # Create dataloaders
        train_loaders = [
            DataLoader(
                DatasetSubset(train_dataset, dict_users[i]), 
                batch_size=self.batch_size, 
                shuffle=True
            ) for i in range(self.n_clients)
        ]
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loaders, val_loader, test_loader, 3  # 3 channels for CIFAR10