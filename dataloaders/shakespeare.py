import os
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from .base import BaseDataLoader, DatasetSubset

class ShakespeareDataset(Dataset):
    def __init__(self, data_dir, train=True, iid=False, num_clients=None):
        super().__init__()
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        # Read data first
        self.train_clients, _, self.train_data, self.test_data = self._read_data(train_dir, test_dir)
        self.train = train
        self.iid = iid
        self.num_clients = num_clients if num_clients else len(self.train_clients)
        
        # Create vocabulary before preparing data
        self._create_vocabulary()
        
        # Prepare data after vocabulary is created
        if self.train:
            self.data, self.labels, self.client_indices = self._prepare_training_data()
        else:
            self.data, self.labels = self._prepare_test_data()

    def _create_vocabulary(self):
        """Create vocabulary from all available data (train and test)"""
        all_chars = set()
        
        # Collect characters from training data
        for client in self.train_clients:
            # Add characters from input sequences
            if self.train_data[client] is not None:
                all_chars.update(''.join(self.train_data[client]['x']))
                # Add characters from labels
                all_chars.update(''.join(str(c) for c in self.train_data[client]['y']))
            
            # Add characters from test data
            if self.test_data[client] is not None:
                all_chars.update(''.join(self.test_data[client]['x']))
                all_chars.update(''.join(str(c) for c in self.test_data[client]['y']))
        
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
        """Now only handles IID split"""
        all_data = []
        all_labels = []
        
        # Collect all data from all clients
        for client in self.train_clients:
            if self.train_data[client] is not None:
                all_data.extend(self.train_data[client]['x'])
                all_labels.extend(self.train_data[client]['y'])
                
        # Shuffle data
        indices = list(range(len(all_data)))
        np.random.shuffle(indices)
        
        # Split data evenly among clients
        samples_per_client = len(indices) // self.num_clients
        remaining_samples = len(indices) % self.num_clients
        client_indices = {}
        
        current_idx = 0
        for i in range(self.num_clients):
            # Distribute remaining samples evenly
            extra_sample = 1 if i < remaining_samples else 0
            client_size = samples_per_client + extra_sample
            
            end_idx = current_idx + client_size
            client_indices[i] = set(range(current_idx, end_idx))
            current_idx = end_idx
                
        return [all_data[i] for i in indices], [all_labels[i] for i in indices], client_indices
    
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

class ShakespeareDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, n_clients: int, batch_size: int, iid: bool = True):
        super().__init__(data_dir, n_clients, batch_size, iid)

    def load_data(self):
        # Load datasets
        train_dataset = ShakespeareDataset(
            data_dir=self.data_dir,
            train=True,
            iid=self.iid,
            num_clients=self.n_clients
        )
        test_dataset = ShakespeareDataset(
            data_dir=self.data_dir,
            train=False
        )

        # Split test into validation and test
        test_size = len(test_dataset)
        val_size = int(0.2 * test_size)
        test_size = test_size - val_size
        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

        # Create dataloaders using client indices from dataset
        train_loaders = [
            DataLoader(
                DatasetSubset(train_dataset, train_dataset.client_indices[i]),
                batch_size=self.batch_size,
                shuffle=True
            ) for i in range(self.n_clients)
        ]
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loaders, val_loader, test_loader, None  # None for text data