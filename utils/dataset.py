import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from typing import List, Tuple, Optional, Union, Callable

class TiffVolumeDataset(Dataset):
    def __init__(
        self,
        tiff_paths: List[Union[str, Path]],
        subvolume_shape: Tuple[int, int, int],
        stride: Optional[Tuple[int, int, int]] = None,
        transform: Optional[Callable] = None,
        preload: bool = False
    ):
        """
        Dataset for loading 3D volumes from TIFF files and extracting subvolumes.
        
        Args:
            tiff_paths: List of paths to TIFF files
            subvolume_shape: Shape of subvolumes to extract (depth, height, width)
            stride: Step size for sliding window (depth, height, width). If None, uses subvolume_shape
            transform: Transformations to apply to each subvolume
            preload: Whether to load all volumes into memory at initialization
        """
        self.tiff_paths = [Path(p) for p in tiff_paths]
        self.subvolume_shape = subvolume_shape
        self.stride = stride if stride else subvolume_shape
        self.transform = transform
        self.preload = preload
        
        # Store volume metadata and build subvolume index
        self.volumes = []
        self.subvolume_indices = []
        
        for tiff_idx, tiff_path in enumerate(self.tiff_paths):
            # Load volume metadata without loading the entire volume
            volume_info = self._get_volume_info(tiff_path)
            self.volumes.append({
                'path': tiff_path,
                'shape': volume_info['shape'],
                'data': self._load_volume(tiff_path) if preload else None
            })
            
            # Calculate valid subvolume positions
            depth, height, width = volume_info['shape']
            depth_stride, height_stride, width_stride = self.stride
            
            for d in range(0, depth - self.subvolume_shape[0] + 1, depth_stride):
                for h in range(0, height - self.subvolume_shape[1] + 1, height_stride):
                    for w in range(0, width - self.subvolume_shape[2] + 1, width_stride):
                        self.subvolume_indices.append({
                            'volume_idx': tiff_idx,
                            'position': (d, h, w)
                        })
    
    def _get_volume_info(self, tiff_path):
        """Get volume shape without loading entire volume"""
        with Image.open(tiff_path) as img:
            frames = []
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(len(frames))
            except EOFError:
                pass
            
            # Get shape as (depth, height, width)
            if frames:
                shape = (len(frames), frames[0].shape[0], frames[0].shape[1])
            else:
                shape = (0, 0, 0)
                
        return {'shape': shape}
    
    def _load_volume(self, tiff_path):
        """Load entire volume from TIFF file"""
        frames = []
        with Image.open(tiff_path) as img:
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(len(frames))
            except EOFError:
                pass
        
        # Stack frames to create volume
        return np.stack(frames) if frames else np.array([])
    
    def _get_subvolume(self, volume_idx, position):
        """Extract subvolume at specified position"""
        d_start, h_start, w_start = position
        d_end = d_start + self.subvolume_shape[0]
        h_end = h_start + self.subvolume_shape[1]
        w_end = w_start + self.subvolume_shape[2]
        
        if self.preload and self.volumes[volume_idx]['data'] is not None:
            # Extract from preloaded data
            volume_data = self.volumes[volume_idx]['data']
            subvolume = volume_data[d_start:d_end, h_start:h_end, w_start:w_end]
        else:
            # Load frames on-the-fly
            tiff_path = self.volumes[volume_idx]['path']
            frames = []
            with Image.open(tiff_path) as img:
                for i in range(d_start, d_end):
                    img.seek(i)
                    frame = np.array(img)[h_start:h_end, w_start:w_end]
                    frames.append(frame)
            
            subvolume = np.stack(frames)
        
        return subvolume
    
    def __len__(self):
        return len(self.subvolume_indices)
    
    def __getitem__(self, idx):
        # Get subvolume index information
        subvol_info = self.subvolume_indices[idx]
        volume_idx = subvol_info['volume_idx']
        position = subvol_info['position']
        
        # Extract subvolume
        subvolume = self._get_subvolume(volume_idx, position)
        
        # Convert to tensor
        subvolume = torch.from_numpy(subvolume).float()
        
        # Add channel dimension if needed (C, D, H, W)
        if len(subvolume.shape) == 3:
            subvolume = subvolume.unsqueeze(0)
        
        # Apply transformations if specified
        if self.transform:
            subvolume = self.transform(subvolume)
        
        return {
            'volume': subvolume,
            'volume_idx': volume_idx,
            'position': position,
            'path': str(self.volumes[volume_idx]['path'])
        }

class NormalizeVolume:
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, volume):
        if self.min_val is None or self.max_val is None:
            min_val = volume.min()
            max_val = volume.max()
        else:
            min_val = self.min_val
            max_val = self.max_val
        
        volume = (volume - min_val) / (max_val - min_val + 1e-8)
        return volume

class ResizeVolume:
    def __init__(self, target_shape):
        self.target_shape = target_shape
    
    def __call__(self, volume):
        # Assumes volume is (C, D, H, W)
        c, d, h, w = volume.shape
        target_d, target_h, target_w = self.target_shape
        
        # Create resizing transform for spatial dimensions
        resize = T.Resize((target_h, target_w), antialias=True)
        
        # Process each depth slice
        resized_slices = []
        for i in range(d):
            slice_2d = volume[:, i, :, :]
            resized_slice = resize(slice_2d)
            resized_slices.append(resized_slice)
        
        # Stack slices and interpolate depth if needed
        resized_volume = torch.stack(resized_slices, dim=1)
        
        if d != target_d:
            # Interpolate along depth dimension
            resized_volume = torch.nn.functional.interpolate(
                resized_volume, 
                size=(target_d, target_h, target_w), 
                mode='trilinear', 
                align_corners=False
            )
        
        return resized_volume