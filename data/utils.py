from torch.utils.data import Dataset, DataLoader
import scipy
import numpy as np
import jax.numpy as jnp

def load_mat_data(path):
    """
    Loads the mat data from path into a dictionary
    """
    return scipy.io.loadmat(path)

def jax_collate(batch):
    """
    Collate function that converts a batch of numpy arrays to a batch of jax arrays. Use with torch DataLoader.
    """
    if isinstance(batch[0], np.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [jax_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)

class MatDataset(Dataset):
    """
    Load Matlab data, get items according to keys.

    Args:
        path: str
            The path to the .mat file.
        keys: list[str]
            The keys to load from the .mat file.
        transform: callable, optional
            A function to transform the data. Default is None.
    """

    def __init__(self, path, keys, transform=None):
        self.data = load_mat_data(path)
        self.keys = keys
        self.transform = transform

    def __len__(self):
        return len(self.data[self.keys[0]])

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(*[self.data[key][idx] for key in self.keys])
        return tuple([self.data[key][idx] for key in self.keys])

