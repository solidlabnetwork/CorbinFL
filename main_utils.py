# main_utils.py
import torch
import torch.nn as nn
import numpy as np
import random
import GPUtil

def setup_device(device_type: str):
    if device_type == "GPU":
        return select_free_gpu(max_memory_usage=0.5)
    return torch.device("cpu")

def select_free_gpu(max_memory_usage=0.5):
    # Get the list of GPUs
    devices = GPUtil.getGPUs()
    available_gpus = [i for i in range(len(devices)) if devices[i].memoryUtil < max_memory_usage]
    if len(available_gpus) ==0:
        print(f"No available GPU with memory usage < {max_memory_usage}")
        selected_gpu = sorted(devices, key=lambda x: (x.memoryUtil, x.load))[0].id

    else: 
        available_gpu_loads = [devices[i].load for i in available_gpus]
        print(f"Available GPUs with memory usage < {int(max_memory_usage*100)}%: {available_gpus} ")
        print(f"GPU loads: {available_gpu_loads}")
        min_load = min(available_gpu_loads)
        min_load_index = available_gpu_loads.index(min_load)
        selected_gpu = available_gpus[min_load_index]
    print(f"Selected GPU: {selected_gpu} (memory = {devices[selected_gpu].memoryUtil*100}%, load  {devices[selected_gpu].load*100}%)")
            # Set the selected GPU as the default device
    device = torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")
    return device

def validate(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




