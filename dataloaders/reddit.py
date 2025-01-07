import os
import json
import torch
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Set
from .base import BaseDataLoader, DatasetSubset

class RedditDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        vocab_path: str,
        split: str = 'train',
        seq_length: int = 10,  # Default to 10 as per data format
        iid: bool = True,
        num_clients: Optional[int] = None
    ):
        """
        Reddit dataset implementation supporting federated learning.
        
        Args:
            data_dir: Directory containing reddit_{split}.json files
            vocab_path: Path to reddit_vocab.pck
            split: 'train', 'test', or 'val'
            seq_length: Sequence length (default 10 based on data format)
            iid: Whether to use IID data distribution
            num_clients: Number of clients for federated setup (only used in train)
        """
        super().__init__()
        self.seq_length = seq_length
        self.split = split
        self.iid = iid
        self.train = split == 'train'
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_dict = pickle.load(f)
            self.vocab = vocab_dict['vocab']
            self.pad_idx = vocab_dict['pad_symbol']
            self.unk_idx = vocab_dict['unk_symbol']
            self.vocab_size = len(self.vocab)

        # Load data and create client splits if needed
        self.sequences, self.client_indices = self._load_and_prepare_data(
            data_dir, 
            num_clients if self.train else None
        )

    def _load_and_prepare_data(
        self, 
        data_dir: str,
        num_clients: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], Optional[Dict[int, Set[int]]]]:
        """Load and prepare data from JSON files"""
        sequences = []
        json_files = [f for f in os.listdir(data_dir) 
                     if f.endswith(f'_{self.split}.json')]
        json_files.sort()

        # Process each file
        for json_file in json_files:
            with open(os.path.join(data_dir, json_file)) as f:
                data = json.load(f)
                user_data = data['user_data']

                # Process each user's sequences
                for user_id, user_content in user_data.items():
                    for sequence in user_content['x']:
                        # Convert tokens to indices
                        token_indices = [
                            self.vocab.get(token, self.unk_idx) 
                            for token in sequence
                        ]
                        sequences.append(torch.tensor(token_indices))

        # Create client splits for training data
        client_indices = None
        if self.train and num_clients is not None:
            if self.iid:
                # IID split: randomly distribute sequences among clients
                indices = list(range(len(sequences)))
                np.random.shuffle(indices)
                
                samples_per_client = len(indices) // num_clients
                remaining_samples = len(indices) % num_clients
                
                client_indices = {}
                current_idx = 0
                
                for i in range(num_clients):
                    extra_sample = 1 if i < remaining_samples else 0
                    client_size = samples_per_client + extra_sample
                    
                    end_idx = current_idx + client_size
                    client_indices[i] = set(indices[current_idx:end_idx])
                    current_idx = end_idx
            else:
                # Non-IID split: maintain user-based grouping
                raise NotImplementedError("Non-IID splitting not implemented yet")

        return sequences, client_indices

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input sequence and target for language modeling"""
        sequence = self.sequences[idx]
        
        # Input is all tokens except last
        x = sequence[:-1]
        # Target is all tokens except first
        y = sequence[1:]
        
        return x, y

    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size

    def decode_sequence(self, indices: torch.Tensor) -> List[str]:
        """Convert a sequence of indices back to tokens"""
        idx_to_token = {idx: token for token, idx in self.vocab.items()}
        return [idx_to_token.get(idx.item(), '<UNK>') for idx in indices]

    def create_padding_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        """Create padding mask for transformer attention"""
        return (sequence == self.pad_idx)

class RedditDataLoader(BaseDataLoader):
    def __init__(
        self, 
        data_dir: str,
        n_clients: int,
        batch_size: int,
        seq_length: int = 10,
        iid: bool = True
    ):
        """
        Reddit data loader inheriting from BaseDataLoader.
        
        Args:
            data_dir: Root directory containing train/val/test subdirectories
            vocab_path: Path to vocabulary file
            n_clients: Number of federated clients
            batch_size: Batch size for dataloaders
            seq_length: Sequence length (default 10 based on data format)
            iid: Whether to use IID data distribution
        """
        super().__init__(data_dir, n_clients, batch_size, iid)
        self.vocab_path = data_dir + '/reddit_vocab.pck'
        self.seq_length = seq_length

    def load_data(self) -> Tuple[List[DataLoader], DataLoader, DataLoader, int]:
        """
        Load and prepare the dataset.
        
        Returns:
            - List of training DataLoaders (one per client)
            - Validation DataLoader
            - Test DataLoader
            - Number of input channels/features (vocab size in this case)
        """
        # Load datasets
        train_dataset = RedditDataset(
            data_dir=os.path.join(self.data_dir, 'reddit_leaf/train'),
            vocab_path=self.vocab_path,
            split='train',
            seq_length=self.seq_length,
            iid=self.iid,
            num_clients=self.n_clients
        )

        val_dataset = RedditDataset(
            data_dir=os.path.join(self.data_dir, 'reddit_leaf/val'),
            vocab_path=self.vocab_path,
            split='val',
            seq_length=self.seq_length
        )

        test_dataset = RedditDataset(
            data_dir=os.path.join(self.data_dir, 'reddit_leaf/test'),
            vocab_path=self.vocab_path,
            split='test',
            seq_length=self.seq_length
        )

        # Create client-specific training dataloaders
        train_loaders = [
            DataLoader(
                DatasetSubset(train_dataset, train_dataset.client_indices[i]),
                batch_size=self.batch_size,
                shuffle=True
            ) for i in range(self.n_clients)
        ]

        # Create validation and test loaders
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loaders, val_loader, test_loader, train_dataset.get_vocab_size()