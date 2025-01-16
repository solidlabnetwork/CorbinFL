import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .base import BaseDataLoader, DatasetSubset
from collections import OrderedDict
import json
from typing import Dict, Any, Tuple
import ijson
import pickle


class StreamingFEMNISTManager:
    def __init__(self, json_dir: str, cache_size: int = 1000):
        self.json_dir = json_dir
        self.index = {}  
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
        # Check for existing index file
        index_path = os.path.join(json_dir, 'user_index.json')
        if os.path.exists(index_path):
            print("Loading existing index file...")
            with open(index_path, 'r') as f:
                self.index = json.load(f)
        else:
            print("Creating new index file...")
            self._create_index()
            # Save the index
            with open(index_path, 'w') as f:
                json.dump(self.index, f)

    def _create_index(self):
        """Create index of user positions with data validation"""
        valid_users = 0
        skipped_users = 0
        
        for file_name in os.listdir(self.json_dir):
            if not file_name.endswith('.json'):
                continue
                
            file_path = os.path.join(self.json_dir, file_name)
            print(f"Loading users from file: {file_path}")
            
            # First pass: get users and num_samples
            with open(file_path, 'r') as f:
                data = json.load(f)
                users = data['users']
                num_samples = data['num_samples']
            
            # Second pass: find exact positions and validate data
            with open(file_path, 'r') as f:
                data_str = f.read()
            
            user_data_start = data_str.find('"user_data"')
            
            for i, user in enumerate(users):
                # Skip users with zero samples
                if num_samples[i] == 0:
                    print(f"Skipping user {user}: zero samples")
                    skipped_users += 1
                    continue
                    
                user_marker = f'"{user}":'
                user_start = data_str.find(user_marker, user_data_start)
                
                if user_start == -1:
                    print(f"Skipping user {user}: could not find start position")
                    skipped_users += 1
                    continue
                
                # Find end position
                end_pos = len(data_str)
                if i < len(users) - 1:
                    next_user = users[i + 1]
                    next_marker = f'"{next_user}":'
                    next_pos = data_str.find(next_marker, user_start + len(user_marker))
                    if next_pos != -1:
                        end_pos = next_pos
                
                # Quick validation of user data
                try:
                    # Read a small portion of user data to validate
                    data_chunk = data_str[user_start:min(user_start + 1000, end_pos)]
                    if '"x":[]' in data_chunk or '"y":[]' in data_chunk:
                        print(f"Skipping user {user}: empty data arrays")
                        skipped_users += 1
                        continue
                    
                    # Add to index only if validation passes
                    self.index[user] = {
                        'start': user_start,
                        'end': end_pos,
                        'num_samples': num_samples[i],
                        'file': file_path
                    }
                    valid_users += 1
                    
                except Exception as e:
                    print(f"Skipping user {user}: {str(e)}")
                    skipped_users += 1
                    continue
        
        print(f"\nIndexing completed:")
        print(f"Valid users: {valid_users}")
        print(f"Skipped users: {skipped_users}")
        
        # Save index for future use
        index_path = os.path.join(self.json_dir, 'user_index.pkl')
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)

    def _parse_user_data(self, file_handle, user_pos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse user data using precise position information
        """
        try:
            file_handle.seek(user_pos['start'])
            data_chunk = file_handle.read(user_pos['end'] - user_pos['start'])
            
            # Find the actual data boundaries
            start_brace = data_chunk.find('{')
            if start_brace == -1:
                print("No starting brace found in chunk")
                return {'x': [], 'y': []}
                
            # Find the matching closing brace
            brace_count = 0
            end_brace = -1
            for i, char in enumerate(data_chunk[start_brace:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_brace = start_brace + i + 1
                        break
            
            # Extract just the valid JSON object
            json_str = data_chunk[start_brace:end_brace]
            
            try:
                user_data = json.loads(json_str)
                
                # The data is directly in the object, no need to look for user_id
                if 'x' not in user_data or 'y' not in user_data:
                    print(f"Missing x or y data")
                    return {'x': [], 'y': []}
                    
                return {
                    'x': user_data['x'],
                    'y': user_data['y']
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return {'x': [], 'y': []}
                
        except Exception as e:
            print(f"General error in parse_user_data: {str(e)}")
            return {'x': [], 'y': []}
                

    def _trim_cache(self):
        """Trim cache using LRU strategy"""
        target_size = int(0.9 * self.cache_size)
        while len(self.cache) > target_size:
            self.cache.popitem(last=False)
            
    def _update_cache(self, user_id: str, data: Dict[str, Any]):
        """Update cache with new user data"""
        if user_id in self.cache:
            self.cache.pop(user_id)
        self.cache[user_id] = data
        if len(self.cache) > self.cache_size:
            self._trim_cache()

    def load_users_data(self, selected_users):
        """Load user data with improved efficiency"""
        user_data = {}
        file_groups = {}
        
        # First check cache
        for user in selected_users:
            if user in self.cache:
                data = self.cache.pop(user)
                self.cache[user] = data
                user_data[user] = data
                continue
            
            # Group uncached users by file
            file_path = self.index[user]['file']
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(user)
        
        # Load uncached users by file
        for file_path, users in file_groups.items():
            # print(f"Loading users from file: {file_path}")
            with open(file_path, 'r') as f:
                for user in users:
                    user_pos = self.index[user]
                    data = self._parse_user_data(f, user_pos)
                    user_data[user] = data
                    self._update_cache(user, data)
        
        return user_data

# The FEMNISTDataset class remains unchanged
class FEMNISTDataset(Dataset):
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
            # img = self.transform(img)
            img = img # no transformation needed
            
        return img, target



class FEMNISTDataLoader(BaseDataLoader):
    def __init__(self, data_dir: str, n_clients: int, batch_size: int):
        super().__init__(data_dir, n_clients, batch_size)
        self.transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.json_dir = os.path.join(data_dir, "all_data")
        self.streaming_manager = StreamingFEMNISTManager(self.json_dir)

    def load_data(self):
        """Load and prepare the dataset with non-IID distribution and error handling"""
        # Keep track of failed users
        failed_users = set()
        max_retries = 10  # Maximum number of times to retry with different users
        
        for attempt in range(max_retries):
            try:
                # Select random clients for this round, excluding failed users
                available_users = [user for user in self.streaming_manager.index.keys() 
                                if user not in failed_users]
                
                if len(available_users) < self.n_clients:
                    raise ValueError(f"Not enough valid users left. Only {len(available_users)} available")
                    
                selected_users = np.random.choice(available_users, self.n_clients, replace=False)
                
                # Load selected users' data
                user_data = self.streaming_manager.load_users_data(selected_users)
                
                # Create datasets and prepare for weight calculation
                train_loaders = []
                train_sizes = []
                val_data = {'x': [], 'y': []}
                test_data = {'x': [], 'y': []}
                
                for user in selected_users:
                    data = user_data[user]
                    if not data or len(data.get('y', [])) == 0:
                        print(f"Empty data for user {user}, skipping...")
                        failed_users.add(user)
                        raise ValueError(f"Empty data for user {user}")
                        
                    n_samples = len(data['y'])
                    if n_samples == 0:
                        print(f"Zero samples for user {user}, skipping...")
                        failed_users.add(user)
                        raise ValueError(f"Zero samples for user {user}")
                    
                    # Create copies for training data
                    train_data = {'x': data['x'].copy(), 'y': data['y'].copy()}
                    
                    # Take samples for validation and test if needed
                    if len(val_data['x']) < 1000:
                        val_size = min(int(0.1 * n_samples), 100)
                        val_indices = np.random.choice(n_samples, val_size, replace=False)
                        for idx in sorted(val_indices, reverse=True):
                            val_data['x'].append(train_data['x'].pop(idx))
                            val_data['y'].append(train_data['y'].pop(idx))
                            
                    if len(test_data['x']) < 1000:
                        test_size = min(int(0.1 * n_samples), 100)
                        test_indices = np.random.choice(len(train_data['y']), test_size, replace=False)
                        for idx in sorted(test_indices, reverse=True):
                            test_data['x'].append(train_data['x'].pop(idx))
                            test_data['y'].append(train_data['y'].pop(idx))
                    
                    # Verify we still have training data after val/test split
                    if len(train_data['y']) == 0:
                        print(f"No training data left after splits for user {user}, skipping...")
                        failed_users.add(user)
                        raise ValueError(f"No training data left for user {user}")
                    
                    # Create dataset with remaining training data
                    train_dataset = FEMNISTDataset(train_data, self.transform)
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        shuffle=True
                    )
                    train_loaders.append(train_loader)
                    train_sizes.append(len(train_data['y']))
                
                # If we got here, all users were processed successfully
                # Calculate weights and create val/test datasets
                total_train_samples = sum(train_sizes)
                client_weights = [size / total_train_samples for size in train_sizes]
                
                val_dataset = FEMNISTDataset(val_data, self.transform)
                test_dataset = FEMNISTDataset(test_data, self.transform)
                
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                
                return train_loaders, val_loader, test_loader, client_weights, 1
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to load data after {max_retries} attempts")
                continue

    @property
    def client_weights(self):
        """Get the weights of currently selected clients"""
        if hasattr(self, '_client_weights'):
            return self._client_weights
        return None
    