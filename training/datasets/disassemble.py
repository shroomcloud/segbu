import numpy as np


def disassebmle_w_mask(img: np.ndarray, mask: np.ndarray, res: int, overlap: float):
    """Break soucre image and gt mask into tiles of specified resolution with overlap.

    Args:
        img (np.ndarray): Source image.
        mask (np.ndarray): Ground truth.
        res (int): Tile resolution.
        overlap (float): [0; 1] overlap of two neighbouring tiles.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Pairs of tiles image-mask.
    """
    stride = res - int(res * overlap)
    h, w, _ = img.shape
    tiles = []
    for y in range(0, h - (h % stride), stride):
        for x in range(0, w - (w % stride), stride):
            if (h - y) < res or (w - x) < res:
                break
            tile_img, tile_msk = (
                img[:, y : y + res, x : x + res],
                mask[:, y : y + res, x : x + res],
            )
            tiles.append((tile_img, tile_msk))

    if w % stride != 0:
        for y in range(0, h - (h % stride), stride):
            if (h - y) < res:
                break
            tile_img, tile_msk = (
                img[:, y : y + res, w - res : w],
                mask[:, y : y + res, w - res : w],
            )
            tiles.append((tile_img, tile_msk))

    if h % stride != 0:
        for x in range(0, w - (w % stride), stride):
            if (w - x) < res:
                break
            tile_img, tile_msk = (
                img[:, h - res : h, x : x + res],
                mask[:, h - res : h, x : x + res],
            )
            tiles.append((tile_img, tile_msk))

    if h % stride != 0 and w % stride != 0:
        tile_img, tile_msk = (
            img[:, h - res : h, w - res : w],
            mask[:, h - res : h, w - res : w],
        )
        tiles.append((tile_img, tile_msk))

    return tiles
