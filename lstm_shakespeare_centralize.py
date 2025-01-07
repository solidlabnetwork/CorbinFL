# train_lstm_centralize.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Adjust these imports to match your project structure
# (e.g. from yourmodule.dataloaders.shakespeare import ShakespeareDataLoader)
from dataloaders.shakespeare import ShakespeareDataLoader 
from Nets import CharLSTM   # or wherever your CharLSTM is defined

def train_lstm_centralize(
    data_dir: str,
    epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = 'cpu'
):
    """
    Train a CharLSTM model on the entire Shakespeare dataset
    as if it's centralized (only one client).
    
    Args:
        data_dir (str): Path to directory containing 'train/' and 'test/' JSON data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (str): 'cpu' or 'cuda'.
    """
    # 1) Load data with exactly 1 "client" to mimic centralized learning
    print("Loading data...")
    data_loader = ShakespeareDataLoader(data_dir=data_dir, n_clients=1, batch_size=batch_size, iid=True)
    train_loaders, val_loader, test_loader, _ = data_loader.load_data()
    
    # Since n_clients=1, we only have train_loaders[0]
    train_loader = train_loaders[0]
    
    # 2) Initialize the CharLSTM model
    model = CharLSTM()
    model.to(device)
    
    # 3) Define optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Containers to store metrics for plotting
    train_losses = []
    test_accuracies = []
    
    # 4) Training loop
    print("Starting training...")
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)       # shape: [batch_size, 80]
            loss = criterion(outputs, targets)  # single-label classification
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss over the epoch
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # 5) Evaluate on test set
        test_acc = evaluate(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch}/{epochs}: "
              f"Train Loss = {epoch_loss:.4f}, "
              f"Test Accuracy = {test_acc:.2f}%")
    
    # 6) Plot the results
    plot_results(train_losses, test_accuracies)

def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str):
    """
    Evaluate the model on the given data loader and return accuracy.
    
    Args:
        model (nn.Module): Trained model
        data_loader (DataLoader): Loader with test (or val) data
        device (str): 'cpu' or 'cuda'
    
    Returns:
        float: Accuracy in percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # shape [B, 80]
            preds = outputs.argmax(dim=1)  # pick class with highest logit
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0

def plot_results(train_losses, test_accuracies):
    """
    Plot the training loss and test accuracy over epochs.
    
    Args:
        train_losses (List[float]): List of average training losses per epoch
        test_accuracies (List[float]): List of test accuracies per epoch
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, marker='o', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Feel free to modify these or parse them from sys.argv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_directory = "./data/Shakespeare"  # your path to Shakespeare data
    
    train_lstm_centralize(
        data_dir=data_directory,
        epochs=20,
        lr=0.001,
        batch_size=64,
        device=device
    )
