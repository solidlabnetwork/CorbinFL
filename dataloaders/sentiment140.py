#source URL:
# https://huggingface.co/datasets/stanfordnlp/sentiment140/blob/main/sentiment140.py
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from .base import BaseDataLoader, DatasetSubset

class Sent140Dataset(Dataset):
    def __init__(self, data_dir, train=True, iid=False, num_clients=None, seq_len=25):
        super().__init__()
        self.data_dir = data_dir
        self.train = train
        self.iid = iid
        self.seq_len = seq_len
        
        # Load embeddings first
        self.word_to_idx, self.embeddings = self._load_embeddings()
        self.vocab_size = len(self.word_to_idx)
        self.unk_idx = self.word_to_idx.get('<UNK>', 0)
        
        # Read and process data
        self.train_data, self.test_data = self._read_data()
        self.num_clients = num_clients if num_clients else len(set(self.train_data['user']))
        
        # Prepare data
        if self.train:
            self.data, self.labels, self.client_indices = self._prepare_training_data()
        else:
            self.data, self.labels = self._prepare_test_data()

    def _load_embeddings(self):
        """Load pre-processed embeddings from embs.json"""
        embs_path = os.path.join(self.data_dir, 'embs.json')
        with open(embs_path, 'r') as f:
            embs_data = json.load(f)
        
        # Create word to index mapping
        word_to_idx = {word: idx for idx, word in enumerate(embs_data['vocab'])}
        embeddings = np.array(embs_data['emba'])
        
        return word_to_idx, embeddings

    def _read_data(self):
        """Read CSV data and perform basic preprocessing"""
        data_path = os.path.join(self.data_dir, 'sentiment140', 'training.1600000.processed.noemoticon.csv')
        
        # Read CSV file
        full_data = pd.read_csv(data_path, encoding='latin-1', header=None,
                              names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
        
        # Convert sentiment labels (0=negative, 4=positive) to binary (0=negative, 1=positive)
        full_data['sentiment'] = (full_data['sentiment'] // 4).astype(int)
        
        # Split data into train/test sets (90% train, 10% test)
        np.random.seed(42)  # for reproducibility
        mask = np.random.rand(len(full_data)) < 0.9
        train_data = full_data[mask]
        test_data = full_data[~mask]
        
        return train_data, test_data

    def _tokenize_and_pad(self, text):
        """Convert text to sequence of word indices and pad/truncate to seq_len"""
        # Basic tokenization (split on whitespace)
        tokens = str(text).lower().split()
        
        # Convert tokens to indices
        indices = [self.word_to_idx.get(token, self.unk_idx) for token in tokens]
        
        # Pad or truncate to seq_len
        if len(indices) < self.seq_len:
            indices = indices + [self.unk_idx] * (self.seq_len - len(indices))
        else:
            indices = indices[:self.seq_len]
            
        return indices

    def _prepare_training_data(self):
        """Prepare training data with IID distribution"""
        # Process all texts and labels
        all_data = [self._tokenize_and_pad(text) for text in self.train_data['text']]
        all_labels = self.train_data['sentiment'].tolist()
        
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
        all_data = [self._tokenize_and_pad(text) for text in self.test_data['text']]
        all_labels = self.test_data['sentiment'].tolist()
        return all_data, all_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a single item from the dataset"""
        sequence = torch.tensor(self.data[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return sequence, label

    def get_vocab_size(self):
        """Return the vocabulary size"""
        return self.vocab_size

    def get_embeddings(self):
        """Return the embedding matrix"""
        return torch.FloatTensor(self.embeddings)

class Sent140DataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, n_clients: int, batch_size: int, iid: bool = True):
        super().__init__(data_dir, n_clients, batch_size, iid)

    def load_data(self):
        # Load datasets
        train_dataset = Sent140Dataset(
            data_dir=self.data_dir,
            train=True,
            iid=self.iid,
            num_clients=self.n_clients
        )
        test_dataset = Sent140Dataset(
            data_dir=self.data_dir,
            train=False
        )

        # Split test into validation and test
        test_size = len(test_dataset)
        val_size = int(0.2 * test_size)
        test_size = test_size - val_size
        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

        # Create dataloaders
        train_loaders = [
            DataLoader(
                DatasetSubset(train_dataset, train_dataset.client_indices[i]),
                batch_size=self.batch_size,
                shuffle=True
            ) for i in range(self.n_clients)
        ]
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Return the embedding matrix as additional data
        embeddings = train_dataset.get_embeddings()

        return train_loaders, val_loader, test_loader, embeddings