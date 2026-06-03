import glob
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils import data


class HSIDataset(data.Dataset):
    """
    PyTorch Dataset for hyperspectral images (.mat files).

    Args:
        root_dir (str or Path): Directory containing .mat files.
        img_size (int): Original image size (height) for cropping logic.
        crop_size (int): Target crop height after random cropping.
        width (int): Width of each spatial patch (sliding window width).
        mode (str): 'train' or 'test' (case-insensitive). Determines whether to apply random cropping/flip.
        marginal (int): Number of spectral bands (or number of overlapped patches along width dimension).
        return_min_max (bool): If True, return (image, filename, min_val, max_val).
        recursive (bool): If True, search .mat files recursively in subdirectories.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        img_size: int = 256,
        crop_size: int = 128,
        width: int = 4,
        mode: str = 'train',
        marginal: int = 60,
        return_min_max: bool = False,
        recursive: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.mode = mode.lower()
        self.img_size = img_size
        self.crop_size = crop_size
        self.width = width
        self.marginal = marginal
        self.return_min_max = return_min_max

        # Collect all .mat files in the directory (optionally recursive)
        pattern = '**/*.mat' if recursive else '*.mat'
        self.file_paths = list(self.root_dir.glob(pattern))
        if not self.file_paths:
            raise FileNotFoundError(f"No .mat files found in {self.root_dir}")
        self.n_images = len(self.file_paths)

        # Precompute random shift boundaries for training (used in __getitem__)
        if self.mode == 'train':
            self.shift_h = (self.img_size - self.crop_size) // 2
            self.shift_w = (self.img_size - self.width) // 2 - self.marginal
            if self.shift_h < 0 or self.shift_w < 0:
                raise ValueError(
                    f"Invalid crop dimensions: img_size={img_size}, crop_size={crop_size}, "
                    f"width={width}, marginal={marginal}. Ensure img_size >= crop_size and "
                    f"img_size >= width + 2*marginal."
                )

    def __len__(self) -> int:
        return self.n_images

    def _load_mat(self, file_path: Path) -> np.ndarray:
        """Load .mat file and extract the first variable (assumed to be the HSI data)."""
        mat = loadmat(file_path)
        # The actual data variable is usually the last key (or the one not starting with '__')
        # We replicate the original behaviour: take the last key.
        var_name = list(mat.keys())[-1]
        data_array = mat[var_name].astype(np.float32)
        return data_array

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, str], Tuple[torch.Tensor, str, float, float]]:
        """
        Returns:
            If return_min_max is False: (tensor_image, filename)
            Else: (tensor_image, filename, min_val, max_val)
        """
        file_path = self.file_paths[index]
        try:
            x = self._load_mat(file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

        # Compute global min/max for later normalization (original values)
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmin == xmax:
            # Degenerate case: constant image
            print(f"Warning: {file_path.name} has constant values. Returning zeros.")
            dummy_shape = (self.marginal, self.crop_size if self.mode == 'train' else self.width, self.width, x.shape[-1])
            zeros = torch.zeros(dummy_shape, dtype=torch.float32)
            if self.return_min_max:
                return zeros, str(file_path), xmin, xmax
            return zeros, str(file_path)

        # Training mode: random cropping and flipping
        if self.mode == 'train':
            # Randomly choose top-left corner for cropping
            w_shift = random.randint(0, self.shift_w)
            h_shift = random.randint(0, self.shift_h)
            h_end = h_shift + self.crop_size

            patches = []
            for k in range(self.marginal):
                # Crop a patch: width = self.width, starting at column (w_shift + k)
                patch = x[h_shift:h_end, w_shift + k : w_shift + self.width + k, :]
                # Random horizontal / vertical flip
                flip = random.random()
                if flip < 0.25:
                    patch = patch[::-1, :, :]      # vertical flip
                elif flip < 0.5:
                    patch = patch[:, ::-1, :]      # horizontal flip
                # Convert to torch tensor (copy to avoid negative stride issues)
                patch_tensor = torch.from_numpy(patch.copy())
                patches.append(patch_tensor)
            x_tensor = torch.stack(patches)   # shape: (marginal, crop_size, width, channels)
        else:
            # Test mode: slide with stride = width (non-overlapping windows along width dimension)
            patches = []
            # Note: original code iterates over column steps with step = width
            # The number of patches along width is determined by the image width.
            for start_col in range(0, x.shape[1], self.width):
                patch = x[:, start_col:start_col + self.width, :]
                patch_tensor = torch.from_numpy(patch.copy())
                patches.append(patch_tensor)
            x_tensor = torch.stack(patches)   # shape: (num_patches, ?, width, channels)

        # Normalize to [0, 1] using global min/max of the whole image
        x_tensor = (x_tensor - xmin) / (xmax - xmin)

        if self.return_min_max:
            return x_tensor, str(file_path), xmin, xmax
        return x_tensor, str(file_path)
