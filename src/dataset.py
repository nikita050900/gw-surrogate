"""
Dataset class for the gravitational-waveform surrogate model.

Handles:
  - loading waveform data from HDF5
  - computing normalization statistics from the training set
  - standardising inputs and targets to zero-mean / unit-variance

Why normalise?
  The raw inputs (m1, m2 ~ 5-20 Msun) and outputs (log-amplitude ~ -29,
  phase ~ 0-1400 rad) live on very different scales.  Without
  standardisation the loss would be dominated by whichever quantity has
  the largest numerical values, and gradient descent would be slow and
  unstable.  Standardising to mean=0, std=1 puts every dimension on
  equal footing so that MSE loss treats amplitude and phase errors
  equally.

Usage:
  train_ds = WaveformDataset("data/train.h5")
  val_ds   = WaveformDataset("data/val.h5", norm_stats=train_ds.norm_stats)
  test_ds  = WaveformDataset("data/test.h5", norm_stats=train_ds.norm_stats)

  The validation and test sets MUST use the training set's statistics --
  otherwise you're leaking information about the val/test distribution
  into the normalisation.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    """
    PyTorch Dataset for (m1, m2) -> (log_amplitude, phase) mapping.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file produced by generate_data.py.
    norm_stats : dict or None
        If None (training set), compute statistics from this file.
        If provided (val/test), use these statistics instead.
    """

    def __init__(self, filepath, norm_stats=None):
        with h5py.File(filepath, "r") as f:
            self.params = f["params"][:]            # (N, 2)
            log_amp = f["log_amplitude"][:]         # (N, n_freq)
            phase = f["phase"][:]                   # (N, n_freq)
            self.f_array = f["f_array"][:]          # (n_freq,)

        # concatenate log-amplitude and phase into one target array
        # shape: (N, 2 * n_freq)
        # first 200 columns = log-amplitude, next 200 = phase
        self.targets = np.concatenate([log_amp, phase], axis=1)

        # --- normalisation ---
        if norm_stats is None:
            # compute from this dataset (training set)
            self.norm_stats = {
                "params_mean": self.params.mean(axis=0),
                "params_std":  self.params.std(axis=0),
                "target_mean": self.targets.mean(axis=0),
                "target_std":  self.targets.std(axis=0),
            }
        else:
            self.norm_stats = norm_stats

        # apply standardisation
        self.params_norm = (
            (self.params - self.norm_stats["params_mean"])
            / self.norm_stats["params_std"]
        ).astype(np.float32)

        self.targets_norm = (
            (self.targets - self.norm_stats["target_mean"])
            / self.norm_stats["target_std"]
        ).astype(np.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.params_norm[idx])     # shape (2,)
        y = torch.from_numpy(self.targets_norm[idx])     # shape (400,)
        return x, y

    @property
    def n_freq(self):
        return len(self.f_array)

    @property
    def input_dim(self):
        return self.params.shape[1]     # 2

    @property
    def output_dim(self):
        return self.targets.shape[1]    # 400

    def denormalize_targets(self, y_norm):
        """
        Convert normalised network output back to physical units.

        Parameters
        ----------
        y_norm : numpy array, shape (..., 400)

        Returns
        -------
        log_amp : array, shape (..., 200)
        phase   : array, shape (..., 200)
        """
        y = y_norm * self.norm_stats["target_std"] + self.norm_stats["target_mean"]
        n = self.n_freq
        return y[..., :n], y[..., n:]
