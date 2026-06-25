# main_utils.py
import torch
import torch.nn as nn
import numpy as np
import random
import GPUtil

def setup_device(device_type: str):
    if device_type == "GPU":
        try:
            return select_free_gpu(max_memory_usage=0.5)
        except Exception as e:
            print(f"Error selecting GPU: {e}")
            print("No free GPU available. Selecting CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

def select_free_gpu(max_memory_usage=0.5, priority="memory"):
    """
    Select a GPU whose memory usage is less than `max_memory_usage`.
    Among those, either pick the GPU with the lowest load (priority="load")
    or the one with the lowest memory usage (priority="memory").
    If no GPU satisfies the memory criterion, pick the GPU with the smallest
    (memoryUtil, load) among all GPUs.
    """
    devices = GPUtil.getGPUs()
    # Filter out GPUs that exceed the memory threshold
    available_gpus = [i for i in range(len(devices)) if devices[i].memoryUtil < max_memory_usage]

    if not available_gpus:
        print(f"No available GPU with memory usage < {max_memory_usage*100:.0f}%.")
        # Fallback: pick GPU with smallest memory usage, then load
        selected_gpu = sorted(devices, key=lambda x: (x.memoryUtil, x.load))[0].id
    else:
        print(f"GPUs below {max_memory_usage*100:.0f}% memory usage: {available_gpus}")
        if priority == "memory":
            # Pick the GPU with the *lowest memory usage* among the filtered set
            mem_util_values = [devices[i].memoryUtil for i in available_gpus]
            min_mem = min(mem_util_values)
            min_mem_index = mem_util_values.index(min_mem)
            selected_gpu = available_gpus[min_mem_index]
        else:
            # Priority="load": pick the GPU with the *lowest load*
            gpu_loads = [devices[i].load for i in available_gpus]
            min_load = min(gpu_loads)
            min_load_index = gpu_loads.index(min_load)
            selected_gpu = available_gpus[min_load_index]
    
    chosen = devices[selected_gpu]
    print(f"Selected GPU: {selected_gpu} "
          f"(memory={chosen.memoryUtil*100:.1f}%, "
          f"load={chosen.load*100:.1f}%)")
    
    return torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")

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




