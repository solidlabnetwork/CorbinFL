# FEMNIST Federated Learning Dataloader

This dataloader is designed for Federated Learning experiments using the FEMNIST (Federated EMNIST) dataset, which contains handwritten character data partitioned by writers.

## Dataset Overview

- **Dataset Type**: Handwritten characters (EMNIST derivative)
- **Task Type**: Multi-class classification
- **Number of Classes**: 62 (digits 0-9, lowercase a-z, uppercase A-Z)
- **Image Size**: 28x28 pixels, grayscale
- **Data Format**: Values normalized to [0, 1]

### Data Distribution Characteristics

- **Non-IID Nature**: Data is naturally non-IID as it's partitioned by writers
- **Client Data Size**: Varies significantly between clients
- **Class Distribution**: Highly imbalanced within clients
  - Some classes may have only 1-2 samples
  - Digits (0-9) typically have more samples (7-12 each)
  - Characters (both cases) often have fewer samples

## Directory Structure

```
data_dir/
├── all_data/
│   ├── all_data_0.json
│   ├── all_data_1.json
│   └── ...
```

## Components

### 1. StreamingFEMNISTManager

Manages efficient data loading with memory optimization:
- Creates and maintains an index of user data locations
- Implements LRU caching mechanism
- Streams data from disk to reduce memory usage
- Saves/loads indices for faster subsequent access

### 2. FEMNISTDataset

PyTorch dataset implementation for FEMNIST:
- Handles individual client datasets
- Loads images as torch tensors
- Shapes data to [1, 28, 28] format
- No normalization/transforms by default

### 3. FEMNISTDataLoader

Main dataloader for federated learning:
- Supports random client selection
- Creates train/val/test splits
- Calculates client weights based on data distribution
- Manages batch creation and iteration

## Usage

```python
# Initialize dataloader
data_dir = "path/to/data"
n_clients = 5  # Number of clients per round
batch_size = 32

loader = FEMNISTDataLoader(data_dir, n_clients, batch_size)

# Get data for one federated round
train_loaders, val_loader, test_loader, client_weights, _ = loader.load_data()

# Train on each client's data
for client_idx, train_loader in enumerate(train_loaders):
    # Your training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        # data.shape: [batch_size, 1, 28, 28]
        # target.shape: [batch_size]
        ...

# Evaluate on validation/test set
for data, target in val_loader:
    ...
```

## Important Notes

1. **Memory Management**:
   - Uses streaming to handle large datasets
   - Implements LRU caching for frequently accessed users
   - Creates index file on first run for faster subsequent access

2. **Data Distribution**:
   - Highly non-IID due to writer-based partitioning
   - Significant class imbalance within clients
   - Consider using techniques to handle imbalanced classes

3. **Client Selection**:
   - Random client selection each round
   - Client weights provided for weighted aggregation
   - Number of clients per round is configurable

4. **Dataset Splits**:
   - Training data: Majority of each client's data
   - Validation data: Up to 10% of samples per client
   - Test data: Up to 10% of samples per client

