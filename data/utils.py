"""
source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Dynamical_Neural_Networks/Complete_DNN_2_2.html#Time-to-code-a-Neural-PDE-Solver
"""

import scipy
import numpy as np
import jax.numpy as jnp
import h5py
import torch
from typing import Tuple
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import jax
from jax.tree_util import tree_map
from jaxtyping import ArrayLike, PRNGKeyArray

#function to torch dataloader from the dataset
def create_dataloader(data_string: str, mode: str, nt: int, nx: int, batch_size:int, num_workers:int):
    try:
        dataset = TrajectoryDataset(data_string,mode,nt=nt,nx=nx)
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    except:
        raise Exception("Datasets could not be loaded properly")

    return loader


#Function to format the data in the correct format
def to_coords(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Transforms the coordinates to a tensor X of shape [time, space, 2].
    Args:
        x: spatial coordinates
        t: temporal coordinates
    Returns:
        torch.Tensor: X[..., 0] is the space coordinate (in 2D)
                      X[..., 1] is the time coordinate (in 2D)
    """
    x_, t_ = torch.meshgrid(x, t)
    x_, t_ = x_.T, t_.T
    return torch.stack((x_, t_), -1)

#Helper class to open the .h5 formated file
class TimeWindowDataset(Dataset):
    """
    Loads an HDF5 PDE dataset and samples a window of history and future time points from each trajectory.

    Args:
        path: str, path to the HDF5 file.
        mode: str, the mode to load the dataset in. Can be 'train', 'val', or 'test'.
        nt: int, the number of time points in the trajectory.
        nx: int, the number of spatial points in the trajectory.
        time_history: int, the number of time points in the history.
        time_future: int, the number of time points in the future.

    """

    def __init__(self, path: str, nt: int, nx: int, history_steps: int, future_steps: int, mode: str, 
                 load_all: bool=False):      
        self.nt = nt
        self.nx = nx
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.mode = mode
        self.max_start_time = self.nt - self.history_steps - self.future_steps
        f = h5py.File(path, 'r')
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'
        self.load_all = load_all
        if load_all:
            data = {self.dataset: self.data[self.dataset][:],
                    'x': self.data['x'][:],
                    't': self.data['t'][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.max_start_time * self.data[self.dataset].shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns data items for batched training/validation/testing.
        Args:
            idx: data index
        Returns:
            torch.Tensor: data trajectory used for training/validation/testing
            torch.Tensor: dt
            torch.Tensor: dx
        """
        # Get the trajectory index and the starting time index
        traj_idx = idx // self.max_start_time
        start_time = idx % self.max_start_time

        history = self.data[self.dataset][traj_idx, start_time:start_time + self.history_steps]
        future = self.data[self.dataset][traj_idx, start_time + self.history_steps:
                                         start_time + self.history_steps + self.future_steps]

        dx = self.data['x'][traj_idx, 1] - self.data['x'][traj_idx, 0]
        dt = self.data['t'][traj_idx, 1] - self.data['t'][traj_idx, 0]

        return history, future, dt, dx

class AugmentedTimeWindowDataset(TimeWindowDataset):
    """
    Loads an HDF5 PDE dataset and samples a window of history and future time points from each trajectory. The samples are augmented with the given transformations.

    Args:
        path: str, path to the HDF5 file.
        mode: str, the mode to load the dataset in. Can be 'train', 'val', or 'test'.
        nt: int, the number of time points in the trajectory.
        nx: int, the number of spatial points in the trajectory.
        time_history: int, the number of time points in the history.
        time_future: int, the number of time points in the future.
        list_transforms: list[Callable], a list of transformations to apply to the samples. Each transformation accepts a sample and a parameter and returns a transformed sample.
    """

    def __init__(self, path: str, nt: int, nx: int, history_steps: int, future_steps: int, mode: str, list_transforms: list, load_all: bool=False, *, key: PRNGKeyArray=None):
        super().__init__(path, nt, nx, history_steps, future_steps, mode, load_all)
        self.list_transforms = list_transforms
        self.key = key
        self.num_transforms = len(list_transforms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_idx = idx // self.max_start_time
        start_time = idx % self.max_start_time
        x = self.data['x'][traj_idx]
        t = self.data['t'][traj_idx]
        X, T = np.meshgrid(x, t)
        TX = np.stack([T, X])
        U = self.data[self.dataset][traj_idx]
        *keys, self.key = jax.random.split(self.key, self.num_transforms + 1)
        for transform, key in zip(self.list_transforms, keys):
            U, TX = transform((U, TX), key=key)                   
        history = U[start_time:start_time + self.history_steps]
        future = U[start_time + self.history_steps:start_time + self.history_steps + self.future_steps]
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        return history, future, dt, dx

    @partial(jax.jit, static_argnames=('self', 'transforms'))
    def jitted_augmenter(self, U: ArrayLike, t: ArrayLike, x: ArrayLike, transforms: List[callable], key: PRNGKeyArray):
        X, T = jnp.meshgrid(x, t)
        TX = jnp.stack([T, X])
        *keys , key = jax.random.split(key, len(transforms) + 1)
        for transform, subkey in zip(transforms, keys):
            U, TX = transform(U, TX, key=subkey)
        return U, TX, key
        
class TrajectoryDataset(Dataset):
    """
    Load samples of an PDE Dataset, get items according to PDE.
    """
    def __init__(self, path: str,
                 mode: str,
                 nt: int,
                 nx: int,
                 dtype=np.float64,
                 load_all: bool=False):
        self.nt = nt
        self.nx = nx
        self.mode = mode
        self.dtype = dtype
        f = h5py.File(path, 'r')
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'
        if load_all:
            data = {self.dataset: self.data[self.dataset][:],
                    'x': self.data['x'][:],
                    't': self.data['t'][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns data items for batched training/validation/testing.
        Args:
            idx: data index
        Returns:
            np.ndarray: data trajectory used for training/validation/testing
            np.ndarray: dx
            np.ndarray: dt
        """
        u = self.data[self.dataset][idx]
        x = self.data['x'][idx]
        t = self.data['t'][idx]
        X, T = np.meshgrid(x, t)
        return u, T, X

#function to create x - y data pairs: 20 past timepoints as x, 20 future timepoints as y
def create_data(datapoints: torch.Tensor, start_time: list, time_future: int, time_history: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Getting data of PDEs for training, validation and testing.
    Args:
        datapoints (torch.Tensor): trajectory input
        start_time (int list): list of different starting times for different trajectories in one batch
        pf_steps (int): push forward steps
    Returns:
        torch.Tensor: neural network input data
        torch.Tensor: neural network labels
    """
    data = torch.Tensor()
    labels = torch.Tensor()
    # Loop over batch and different starting points
    # For every starting point, we take the number of time_history points as training data
    # and the number of time future data as labels
    for (dp, start) in zip(datapoints, start_time):
        end_time = start+time_history
        d = dp[start:end_time]
        target_start_time = end_time
        target_end_time = target_start_time + time_future
        l = dp[target_start_time:target_end_time]

        data = torch.cat((data, d[None, :]), 0)
        labels = torch.cat((labels, l[None, :]), 0)

    return data, labels

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

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

