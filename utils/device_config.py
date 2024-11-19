import torch

def configure_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_default_device(device)
    print("Tensors will be allocated to " + str(device))
    return device
