import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2, 40).__str__()
import cv2
import torch
from torch.utils.data import Dataset

import image_utils


class ClassificationDataset(Dataset):

    def __init__(self, image_paths, labels, tile_size, n_tiles, transforms=None):

        self.image_paths = image_paths
        self.labels = labels
        self.tile_size = tile_size
        self.n_tiles = n_tiles
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.image_paths))

        Returns
        -------
        tiles (torch.FloatTensor of shape (n_tiles, 3, height, width)): Tiles tensor
        label (torch.FloatTensor of shape (1)): Label tensor
        """

        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tiles = image_utils.tile_image(
            image=image,
            tile_size=self.tile_size,
            n_tiles=self.n_tiles
        )
        del image

        if self.labels is not None:

            label = self.labels[idx]
            label = torch.as_tensor(label, dtype=torch.float)
            label = torch.unsqueeze(label, dim=0)

            if self.transforms is not None:
                # Apply transforms to tiles and stack them along the batch dimension
                tiles = [self.transforms(image=tile)['image'].float() for tile in tiles]
                tiles = torch.stack(tiles, dim=0)
            else:
                tiles = [torch.as_tensor(tile, dtype=torch.float) for tile in tiles]
                tiles = torch.stack(tiles, dim=0)
                tiles = torch.permute(tiles, dims=(0, 3, 1, 2))
                # Scale pixel values by max 8 bit pixel value
                tiles /= 255.

            return tiles, label

        else:

            if self.transforms is not None:
                # Apply transforms to tiles and stack them along the batch dimension
                tiles = [self.transforms(image=tile)['image'].float() for tile in tiles]
                tiles = torch.stack(tiles, dim=0)
            else:
                tiles = [torch.as_tensor(tile, dtype=torch.float) for tile in tiles]
                tiles = torch.stack(tiles, dim=0)
                tiles = torch.permute(tiles, dims=(0, 3, 1, 2))
                # Scale pixel values by max 8 bit pixel value
                tiles /= 255.

            return tiles
