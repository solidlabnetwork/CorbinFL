
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
        seq_length: int = 10,
        iid: bool = True,
        num_clients: Optional[int] = None,
        client_id: Optional[int] = None,
        clients_root_dir: Optional[str] = None
    ):
        super().__init__()
        self.seq_length = seq_length
        self.split = split
        self.iid = iid
        self.train = split == 'train'
        self.client_id = client_id
        
        # Setup client directory based on number of clients and IID/non-IID
        if clients_root_dir and num_clients:
            distribution_type = "iid" if iid else "non_iid"
            self.clients_data_dir = os.path.join(clients_root_dir, f"{num_clients}_clients_{distribution_type}")
        else:
            self.clients_data_dir = None
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_dict = pickle.load(f)
            self.vocab = vocab_dict['vocab']
            self.pad_idx = vocab_dict['pad_symbol']
            self.unk_idx = vocab_dict['unk_symbol']
            self.vocab_size = len(self.vocab)

        if self.train:
            if client_id is not None:
                # Load specific client data
                self.sequences = self._load_client_data(client_id)
                self.client_indices = None
            else:
                # Check if data for this configuration exists
                if self.clients_data_dir and os.path.exists(self.clients_data_dir):
                    print(f"Using existing client data from {self.clients_data_dir}")
                    self.sequences = []  # Don't load any sequences in memory
                else:
                    # Initial data processing and client split creation
                    self._process_and_save_client_data(data_dir, num_clients)
                    self.sequences = []
        else:
            # For validation and test sets, check if preprocessed data exists
            split_file = os.path.join(self.clients_data_dir, f'{self.split}_data.pt')
            
            if os.path.exists(split_file):
                print(f"Loading preprocessed {self.split} data from disk")
                self.sequences = torch.load(split_file)
            else:
                print(f"Processing {self.split} data and saving to disk")
                self.sequences, _, _ = self._load_and_prepare_data(data_dir)
                # Create directory if it doesn't exist
                os.makedirs(self.clients_data_dir, exist_ok=True)
                # Save processed data
                torch.save(self.sequences, split_file)

    def _process_and_save_client_data(self, data_dir: str, num_clients: int):
        """Process all data and save client-specific datasets to disk"""
        if not self.clients_data_dir:
            raise ValueError("clients_data_dir must be specified for training data")

        # Create clients directory
        os.makedirs(self.clients_data_dir, exist_ok=True)

        # Load all sequences
        sequences, user_ids, _ = self._load_and_prepare_data(data_dir)

        if self.iid:
            # Create IID split
            indices = list(range(len(sequences)))
            np.random.shuffle(indices)
            
            samples_per_client = len(indices) // num_clients
            remaining_samples = len(indices) % num_clients
            
            current_idx = 0
            
            # Save each client's data separately
            for i in range(num_clients):
                extra_sample = 1 if i < remaining_samples else 0
                client_size = samples_per_client + extra_sample
                
                end_idx = current_idx + client_size
                client_indices = indices[current_idx:end_idx]
                
                # Get client's sequences
                client_sequences = [sequences[idx] for idx in client_indices]
                
                # Save client data to disk
                client_path = os.path.join(self.clients_data_dir, f'client_{i}.pt')
                torch.save(client_sequences, client_path)
                
                current_idx = end_idx
        else:
            # Group sequences by user_id for non-IID
            user_to_sequences = {}
            for seq, user_id in zip(sequences, user_ids):
                if user_id not in user_to_sequences:
                    user_to_sequences[user_id] = []
                user_to_sequences[user_id].append(seq)
            
            # Distribute users among clients
            unique_users = list(user_to_sequences.keys())
            np.random.shuffle(unique_users)
            
            users_per_client = len(unique_users) // num_clients
            remaining_users = len(unique_users) % num_clients
            
            current_idx = 0
            
            # Save each client's data
            for i in range(num_clients):
                extra_user = 1 if i < remaining_users else 0
                client_users_count = users_per_client + extra_user
                
                end_idx = current_idx + client_users_count
                client_users = unique_users[current_idx:end_idx]
                
                # Get all sequences for this client's users
                client_sequences = []
                for user_id in client_users:
                    client_sequences.extend(user_to_sequences[user_id])
                
                # Save client data
                client_path = os.path.join(self.clients_data_dir, f'client_{i}.pt')
                torch.save(client_sequences, client_path)
                
                current_idx = end_idx

    def _load_client_data(self, client_id: int) -> List[torch.Tensor]:
        """Load a specific client's data from disk"""
        if not self.clients_data_dir:
            raise ValueError("clients_data_dir must be specified for loading client data")
            
        client_path = os.path.join(self.clients_data_dir, f'client_{client_id}.pt')
        if not os.path.exists(client_path):
            raise ValueError(f"No data found for client {client_id}")
            
        return torch.load(client_path)

    def _load_and_prepare_data(
        self, 
        data_dir: str,
        num_clients: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[str], Optional[Dict[int, Set[int]]]]:
        """Load and prepare data from JSON files"""
        sequences = []
        user_ids = []
        json_files = [f for f in os.listdir(data_dir) 
                     if f.endswith(f'_{self.split}.json')]
        json_files.sort()

        for json_file in json_files:
            with open(os.path.join(data_dir, json_file)) as f:
                data = json.load(f)
                user_data = data['user_data']

                for user_id, user_content in user_data.items():
                    for sequence_list in user_content['x']:
                        if isinstance(sequence_list[0], list):
                            sequence = sequence_list[0]
                        else:
                            sequence = sequence_list

                        token_indices = []
                        for token in sequence:
                            if isinstance(token, str):
                                token_indices.append(self.vocab.get(token, self.unk_idx))
                            else:
                                token_indices.append(self.unk_idx)

                        if len(token_indices) > self.seq_length:
                            token_indices = token_indices[:self.seq_length]
                        elif len(token_indices) < self.seq_length:
                            token_indices.extend([self.pad_idx] * (self.seq_length - len(token_indices)))

                        sequences.append(torch.tensor(token_indices))
                        user_ids.append(user_id)

        return sequences, user_ids, None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        if len(sequence) > self.seq_length + 1:  # +1 for input/target shift
            sequence = sequence[:self.seq_length + 1]
        elif len(sequence) < self.seq_length + 1:
            # Pad sequence to desired length + 1
            padding = torch.full((self.seq_length + 1 - len(sequence),), self.pad_idx)
            sequence = torch.cat([sequence, padding])
        
        x = sequence[:-1]
        y = sequence[1:]
        return x, y

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def decode_sequence(self, indices: torch.Tensor) -> List[str]:
        idx_to_token = {idx: token for token, idx in self.vocab.items()}
        return [idx_to_token.get(idx.item(), '<UNK>') for idx in indices]

class RedditDataLoader(BaseDataLoader):
    def __init__(
        self, 
        data_dir: str,
        n_clients: int,
        batch_size: int,
        seq_length: int = 10,
        iid: bool = True
    ):
        super().__init__(data_dir, n_clients, batch_size, iid)
        self.vocab_path = os.path.join(data_dir, 'reddit_vocab.pck')
        self.seq_length = seq_length
        # Setup client data root directory
        self.clients_root_dir = os.path.join(data_dir, 'client_data')

    def load_data(self) -> Tuple[List[DataLoader], DataLoader, DataLoader, int]:
        """Load and prepare the dataset"""
        # Process training data and save client splits if needed
        train_dataset = RedditDataset(
            data_dir=os.path.join(self.data_dir, 'reddit_leaf/train'),
            vocab_path=self.vocab_path,
            split='train',
            seq_length=self.seq_length,
            iid=self.iid,
            num_clients=self.n_clients,
            clients_root_dir=self.clients_root_dir
        )

        val_dataset = RedditDataset(
            data_dir=os.path.join(self.data_dir, 'reddit_leaf/val'),
            vocab_path=self.vocab_path,
            split='val',
            seq_length=self.seq_length,
            clients_root_dir=self.clients_root_dir,
            num_clients=self.n_clients # to look for the folder related to this number of clients
        )

        test_dataset = RedditDataset(
            data_dir=os.path.join(self.data_dir, 'reddit_leaf/test'),
            vocab_path=self.vocab_path,
            split='test',
            seq_length=self.seq_length,
            clients_root_dir=self.clients_root_dir,
            num_clients=self.n_clients # to look for the folder related to this number of clients
        )

        # Create empty placeholder for train loaders
        train_loaders = [None] * self.n_clients

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

    def get_client_dataloader(self, client_id: int) -> DataLoader:
        """Load a specific client's data and create its DataLoader"""
        distribution_type = "iid" if self.iid else "non_iid"
        client_dataset = RedditDataset(
            data_dir=os.path.join(self.data_dir, 'reddit_leaf/train'),
            vocab_path=self.vocab_path,
            split='train',
            seq_length=self.seq_length,
            iid=self.iid,
            client_id=client_id,
            num_clients=self.n_clients,
            clients_root_dir=self.clients_root_dir
        )

        return DataLoader(
            client_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
