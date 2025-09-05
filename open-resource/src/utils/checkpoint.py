import os, torch
def save_ckpt(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
def load_ckpt(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)
