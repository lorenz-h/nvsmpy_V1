import torch

from gpu_grabber import get_available_gpus

def get_avail_device() -> torch.device:
    gpu_id = get_available_gpus()[0] # select the first gpu
    return torch.device(f"cuda:{gpu_id}")

if __name__ == "__main__":
    
    device = get_avail_device()
    print(device)

    # make sure to manually place all of your tensors on the device we just created
    x = torch.randn(64,1,16,16)
    x.to(device)
    