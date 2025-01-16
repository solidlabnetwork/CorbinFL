import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import os
import json
from typing import List, Tuple
from .base import BaseDataLoader

class IIDFEMNISTDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = []
        self.targets = []
        for x, y in zip(data['x'], data['y']):
            self.data.append(torch.tensor(x).reshape(28, 28).unsqueeze(0))
            self.targets.append(y)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = img  # no transformation needed since data is already normalized
            
        return img, target

class IIDFEMNISTDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, n_clients: int, batch_size: int):
        super().__init__(data_dir, n_clients, batch_size)
        self.transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.json_dir = os.path.join(data_dir, "all_data")
        
    def load_all_data(self):
        """Load and combine all data from JSON files"""
        all_data = {'x': [], 'y': []}
        
        # Iterate through all JSON files in the directory
        for filename in os.listdir(self.json_dir):
            if not filename.endswith('.json') or filename == 'user_index.json':
                continue
                
            filepath = os.path.join(self.json_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Combine data from all users
                for user_id, user_data in data['user_data'].items():
                    all_data['x'].extend(user_data['x'])
                    all_data['y'].extend(user_data['y'])
        
        return all_data

    def load_data(self) -> Tuple[List[DataLoader], DataLoader, DataLoader, List[float], int]:
        """Load and prepare the dataset with IID distribution"""
        try:
            # Load all data
            all_data = self.load_all_data()
            total_samples = len(all_data['y'])
            
            if total_samples == 0:
                raise ValueError("No data loaded")
            
            # Create the complete dataset
            full_dataset = IIDFEMNISTDataset(all_data, self.transform)
            
            # Create validation and test sets (10% each)
            val_size = min(int(0.1 * total_samples), 1000)
            test_size = min(int(0.1 * total_samples), 1000)
            train_size = total_samples - val_size - test_size
            
            # Split indices
            indices = np.random.permutation(total_samples)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create validation and test datasets
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
            
            # Split training data among clients
            samples_per_client = train_size // self.n_clients
            train_loaders = []
            client_weights = []
            
            for i in range(self.n_clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client if i < self.n_clients - 1 else len(train_indices)
                client_indices = train_indices[start_idx:end_idx]
                
                client_dataset = Subset(full_dataset, client_indices)
                client_loader = DataLoader(
                    client_dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                train_loaders.append(client_loader)
                client_weights.append(len(client_indices) / train_size)
            
            # Create validation and test loaders
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Store client weights for later use
            self._client_weights = client_weights
            
            return train_loaders, val_loader, test_loader, client_weights, 1
            
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")

    @property
    def client_weights(self):
        """Get the weights of clients"""
        if hasattr(self, '_client_weights'):
            return self._client_weights
        return None