import numpy as np
import cv2


def resize_with_aspect_ratio(image, longest_edge):

    """
    Resize image while preserving aspect ratio

    Parameters
    ----------
    image (numpy.ndarray of shape (height, width, 3)): Image array
    longest_edge (int): Number of pixels on the longest edge

    Returns
    -------
    image (numpy.ndarray of shape (resized_height, resized_width, 3)): Resized image array
    """

    height, width = image.shape[:2]
    scale = longest_edge / max(height, width)
    image = cv2.resize(image, dsize=(int(width * scale), int(height * scale)), interpolation=cv2.INTER_NEAREST)

    return image


def tile_image(image, tile_size=256, n_tiles=4):

    """
    Resize image while preserving aspect ratio

    Parameters
    ----------
    image (numpy.ndarray of shape (height, width, channel)): Image array
    tile_size (int): Number of pixels on the edges of tiles
    n_tiles (int): Number of tiles

    Returns
    -------
    image (numpy.ndarray of shape (n_tiles, tile_size, tile_size, channel)): Resized image array
    """

    height, width, channel = image.shape
    pad_height, pad_width = (tile_size - height % tile_size) % tile_size, (tile_size - width % tile_size) % tile_size
    padding = [[pad_height // 2, pad_height - pad_height // 2], [pad_width // 2, pad_width - pad_width // 2], [0, 0]]
    image = np.pad(image, padding, mode='constant', constant_values=255)
    image = image.reshape(image.shape[0] // tile_size, tile_size, image.shape[1] // tile_size, tile_size, channel)
    image = image.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, channel)

    if len(image) < n_tiles:
        padding = [[0, n_tiles - len(image)], [0, 0], [0, 0], [0, 0]]
        image = np.pad(image, padding, mode='constant', constant_values=255)

    # Sort tiles by their sums and retrieve top n tiles with the highest sums
    sorting_idx = np.argsort(image.reshape(image.shape[0], -1).sum(-1))[:n_tiles]
    image = image[sorting_idx]

    return image
