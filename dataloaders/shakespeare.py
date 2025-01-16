import os
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from .base import BaseDataLoader, DatasetSubset

class ShakespeareDataset(Dataset):
    def __init__(self, data_dir, train=True, iid=False, num_clients=None, selected_clients=None, client_data=None):
        super().__init__()
        
        if client_data is not None:
            # If client data is provided, use it directly
            self.data = client_data['x']
            self.labels = client_data['y']
            self.train_clients = []
            self.train_data = {}
            self.test_data = {}
        else:
            # Otherwise load from files
            train_dir = os.path.join(data_dir, 'train')
            test_dir = os.path.join(data_dir, 'test')
            
            # Read data first
            self.train_clients, _, self.train_data, self.test_data = self._read_data(train_dir, test_dir)
            self.train = train
            
            # Prepare data
            if self.train:
                self.data, self.labels = self._prepare_training_data()
            else:
                self.data, self.labels = self._prepare_test_data()
        
        # Create vocabulary after data is prepared
        self._create_vocabulary()

    def _create_vocabulary(self):
        """Create vocabulary from all available data (train and test)"""
        all_chars = set()
        
        # Add characters from current data
        all_chars.update(''.join(self.data))
        all_chars.update(''.join(str(c) for c in self.labels))
        
        # Add special tokens
        special_tokens = {'<PAD>', '<UNK>'}
        all_chars.update(special_tokens)
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(special_tokens) + sorted(all_chars - special_tokens))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.pad_idx = self.char_to_idx['<PAD>']
        self.unk_idx = self.char_to_idx['<UNK>']

    def _read_data(self, train_dir, test_dir):
        """Read data from json files"""
        def read_dir(data_dir):
            clients = []
            groups = []
            data = defaultdict(lambda: None)
            
            files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            for f in files:
                file_path = os.path.join(data_dir, f)
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
                clients.extend(cdata['users'])
                if 'hierarchies' in cdata:
                    groups.extend(cdata['hierarchies'])
                data.update(cdata['user_data'])
            
            return sorted(list(set(clients))), groups, data

        train_clients, train_groups, train_data = read_dir(train_dir)
        test_clients, test_groups, test_data = read_dir(test_dir)
        
        return train_clients, train_groups, train_data, test_data

    def _prepare_training_data(self):
        """Prepare training data"""
        all_data = []
        all_labels = []
        for client in self.train_clients:
            if self.train_data[client] is not None:
                all_data.extend(self.train_data[client]['x'])
                all_labels.extend(self.train_data[client]['y'])
        return all_data, all_labels
    
    def _prepare_test_data(self):
        """Prepare test data"""
        all_data = []
        all_labels = []
        for client in self.train_clients:
            if self.test_data[client] is not None:
                all_data.extend(self.test_data[client]['x'])
                all_labels.extend(self.test_data[client]['y'])
        return all_data, all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a single item from the dataset"""
        sequence = self.data[index]
        target = self.labels[index]
        
        # Convert sequence to indices
        if isinstance(sequence, str):
            sequence_indices = torch.tensor([
                self.char_to_idx[c] for c in sequence
            ], dtype=torch.long)
        else:
            sequence_indices = torch.tensor(sequence, dtype=torch.long)
        
        # Convert target to single index
        if isinstance(target, str):
            target_idx = self.char_to_idx[target]
            target_tensor = torch.tensor(target_idx, dtype=torch.long)
        else:
            target_tensor = torch.tensor(target, dtype=torch.long)
        
        return sequence_indices, target_tensor

    def get_vocab_size(self):
        """Return the vocabulary size"""
        return self.vocab_size

    def decode_sequence(self, indices):
        """Convert a sequence of indices back to characters"""
        return ''.join(self.idx_to_char.get(idx.item(), '<UNK>') for idx in indices)


class ShakespeareNonIIDDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, n_clients: int, batch_size: int):
        super().__init__(data_dir, n_clients, batch_size, iid=False)
        
        # Load all data once during initialization
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        # Read all data and store it
        self.train_clients, _, self.train_data, self.test_data = self._read_data(train_dir, test_dir)
        
        # Create and split test dataset once
        self.test_dataset = ShakespeareDataset(
            data_dir=self.data_dir,
            train=False
        )
        
        # Split test into validation and test once
        test_size = len(self.test_dataset)
        val_size = int(0.2 * test_size)
        test_size = test_size - val_size
        self.val_dataset, self.test_dataset = random_split(self.test_dataset, [val_size, test_size])
        
        # Create fixed val and test loaders
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def _read_data(self, train_dir, test_dir):
        """Read data from json files"""
        def read_dir(data_dir):
            clients = []
            groups = []
            data = defaultdict(lambda: None)
            
            files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            for f in files:
                file_path = os.path.join(data_dir, f)
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
                clients.extend(cdata['users'])
                if 'hierarchies' in cdata:
                    groups.extend(cdata['hierarchies'])
                data.update(cdata['user_data'])
            
            return sorted(list(set(clients))), groups, data

        train_clients, train_groups, train_data = read_dir(train_dir)
        test_clients, test_groups, test_data = read_dir(test_dir)
        
        return train_clients, train_groups, train_data, test_data

    def load_data(self):
        """Load data for randomly selected clients"""
        # Randomly select n_clients
        selected_clients = np.random.choice(
            self.train_clients,
            size=min(self.n_clients, len(self.train_clients)),
            replace=False
        ).tolist()
        
        # Create train loaders for selected clients
        train_loaders = []
        client_weights = []
        total_samples = 0
        
        # First pass to calculate total samples
        for client in selected_clients:
            if self.train_data[client] is not None:
                total_samples += len(self.train_data[client]['y'])
        
        # Second pass to create loaders and calculate weights
        for client in selected_clients:
            if self.train_data[client] is not None:
                client_dataset = ShakespeareDataset(
                    data_dir=self.data_dir,
                    train=True,
                    client_data={
                        'x': self.train_data[client]['x'],
                        'y': self.train_data[client]['y']
                    }
                )
                
                train_loaders.append(
                    DataLoader(
                        client_dataset,
                        batch_size=self.batch_size,
                        shuffle=True
                    )
                )
                
                # Calculate weight based on number of samples
                weight = len(self.train_data[client]['y']) / total_samples
                client_weights.append(weight)
        
        return train_loaders, self.val_loader, self.test_loader, client_weights, None