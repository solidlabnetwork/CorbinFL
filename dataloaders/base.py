from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List

class BaseDataLoader(ABC):
    def __init__(self, data_dir: str, n_clients: int, batch_size: int, iid: bool = True):
        self.data_dir = data_dir
        self.n_clients = n_clients
        self.batch_size = batch_size
        self.iid = iid
    
    @abstractmethod
    def load_data(self) -> Tuple[List[DataLoader], DataLoader, DataLoader, int]:
        """
        Load and prepare the dataset
        Returns:
            - List of training DataLoaders (one per client)
            - Validation DataLoader
            - Test DataLoader
            - Number of input channels/features
        """
        pass

class DatasetSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        return self.dataset[data_idx]
    
