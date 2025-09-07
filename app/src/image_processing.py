import numpy as np
import cv2
import tifffile
import io
from dataclasses import dataclass


# image data that needs to be passed from pre to postprocessing
@dataclass(kw_only=True)
class ImageMeta:
    height: int
    width: int
    tile_resolution: int
    src_img: np.ndarray | None = None
    intersection_map: np.ndarray | None = None


def sigmoid(x):
    return 1 / (1 + np.exp(-x) + 1e-8)


# handles images and encodes them into specified format
def handle_and_encode(img: np.ndarray, format_as: str, is_mask: bool = False):
    if is_mask:
        img = (
            np.repeat(img, 3)
            .reshape((img.shape[0], img.shape[1], 3))
            .astype(np.float16)
        )
    if format_as == ".tif":
        if len(img.shape) != 3:
            raise ValueError("Wrong image type")
        bio = io.BytesIO()
        tifffile.imwrite(bio, img)
        bio.seek(0)
        return bio.getvalue()
    else:
        img = img.astype(np.float32)
        if len(img.shape) == 3:
            status, image = cv2.imencode(
                ext=format_as, img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
        else:
            status, image = cv2.imencode(ext=format_as, img=img)
        if status:
            return image
        else:
            raise ValueError("Error while converting image")


# creates an overlayed src/mask image
def overlay_img_mask(src: np.ndarray, mask: np.ndarray):
    src = src.astype(np.uint8)
    mask = np.repeat(mask, 3).reshape((mask.shape[0], mask.shape[1], 3))
    mask = (mask * np.array([255, 64, 64])).astype(np.uint8)
    overlayed = cv2.addWeighted(src, 0.65, mask, 0.35, 0)
    return overlayed


def disassemble(
    img: np.ndarray,
    res: int,
    overlap: float,
    batch_size: int,
    to_resize: int | None = None,
):
    """Tiles an input image and splits the tiles into batches of
    specified size. Also supports tile resizing. Returns map of intersections
    along with tiles.

    Args:
        img (np.ndarray): Source image.
        res (int): Tile resolution.
        overlap (float): [0; 1] overlap of two neighbouring tiles.
        batch_size (int): Batch size
        to_resize (int | None, optional): Resize resolution for tiles. Do not specify to leave the same resolution.

    Returns:
    tuple[list[np.ndarray], np.ndarray]: Pairs of tiles image-mask, intersection map.

    """

    def resize(img_arr: np.ndarray, shape: int, inter=cv2.INTER_AREA):
        native_dtype = img_arr.dtype
        res = cv2.resize(
            np.transpose(img_arr, (1, 2, 0)).astype(np.float32),
            (shape, shape),
            interpolation=inter,
        )
        res = np.transpose(res, (2, 0, 1)).astype(native_dtype)

        return res

    # tiling
    stride = res - int(res * overlap)
    c, h, w = img.shape
    del c
    intersection_map = np.zeros(img.shape[1:])
    tiles = []

    for y in range(0, h - (h % stride), stride):
        for x in range(0, w - (w % stride), stride):
            if (h - y) < res or (w - x) < res:
                break
            tile = img[:, y : y + res, x : x + res]
            tile = resize(tile, to_resize) if to_resize else tile
            intersection_map[y : y + res, x : x + res] += 1
            tiles.append(tile)

    if w % stride != 0:
        for y in range(0, h - (h % stride), stride):
            if (h - y) < res:
                break
            tile = img[:, y : y + res, w - res : w]
            tile = resize(tile, to_resize) if to_resize else tile
            intersection_map[y : y + res, w - res : w] += 1
            tiles.append(tile)

    if h % stride != 0:
        for x in range(0, w - (w % stride), stride):
            if (w - x) < res:
                break
            tile = img[:, h - res : h, x : x + res]
            tile = resize(tile, to_resize) if to_resize else tile
            intersection_map[h - res : h, x : x + res] += 1
            tiles.append(tile)

    if h % stride != 0 and w % stride != 0:
        tile = img[:, h - res : h, w - res : w]
        tile = resize(tile, to_resize) if to_resize else tile
        intersection_map[h - res : h, w - res : w] += 1
        tiles.append(tile)

    # batching
    while batch_size > len(tiles):
        batch_size -= 1

    remainder = len(tiles) % batch_size
    tiles = np.array(tiles)
    if remainder > 0:
        batched = tiles[:-remainder]
        remainder_arr = tiles[-remainder:]
        batched = np.split(batched, len(batched) // batch_size)
        batched.append(remainder_arr)
    else:
        batched = np.split(tiles, len(tiles) // batch_size)

    return batched, intersection_map


def assebmle_logit(batches: list, img_res: tuple, tile_res: int, overlap: float):
    """Assebmles the logit from infered tiles.

    Args:
        batches (list): List of infered batches.
        img_res (tuple): Source image (logit) resolution.
        tile_res (int): Tile resolution.
        overlap (float): [0; 1] overlap of two neighbouring tiles.

    Returns:
        np.ndarray: Stitched logit.
    """
    tiles = list(np.concatenate(batches))
    if len(tiles) == 1:
        return tiles[0]

    h, w = img_res
    stride = tile_res - int(tile_res * overlap)
    result = np.zeros(shape=(h, w))
    to_resize = tile_res != tiles[0].shape[0]

    for y in range(0, h - (h % stride), stride):
        for x in range(0, w - (w % stride), stride):
            if (h - y) < tile_res or (w - x) < tile_res:
                break
            tile = tiles.pop(0)[0]
            if to_resize:
                tile = cv2.resize(
                    tile, (tile_res, tile_res), interpolation=cv2.INTER_NEAREST
                )
            result[y : y + tile_res, x : x + tile_res] += tile

    if w % stride != 0:
        for y in range(0, h - (h % stride), stride):
            if (h - y) < tile_res:
                break
            tile = tiles.pop(0)[0]
            if to_resize:
                tile = cv2.resize(
                    tile, (tile_res, tile_res), interpolation=cv2.INTER_NEAREST
                )
            result[y : y + tile_res, w - tile_res : w] += tile

    if h % stride != 0:
        for x in range(0, w - (w % stride), stride):
            if (w - x) < tile_res:
                break
            tile = tiles.pop(0)[0]
            if to_resize:
                tile = cv2.resize(
                    tile, (tile_res, tile_res), interpolation=cv2.INTER_NEAREST
                )
            result[h - tile_res : h, x : x + tile_res] += tile

    if h % stride != 0 and w % stride != 0:
        tile = tiles.pop(0)[0]
        if to_resize:
            tile = cv2.resize(
                tile, (tile_res, tile_res), interpolation=cv2.INTER_NEAREST
            )
        result[h - tile_res : h, w - tile_res : w] += tile

    return result


# preprocessing logic for model
def unet_preprocessing(
    img: bytes,
    config: dict,
    get_input_img: bool = False,
):
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if len(img.shape) != 3 or (len(img.shape) == 3 and img.shape[2] != 3):
        raise ValueError("Wrong image format")
    elif img.shape[0] < 128 or img.shape[1] < 128:
        raise ValueError("Image too small. Give at least 128x128")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    meta = ImageMeta(
        height=img.shape[0],
        width=img.shape[1],
        tile_resolution=config["tile_res"],
        src_img=img if get_input_img else None,
    )

    mean = np.array(config["norm_mean"])
    std = np.array(config["norm_std"])
    img = (img / 255.0 - mean) / std

    if meta.height <= 384 or meta.width <= 384:
        img = cv2.resize(
            img.astype(np.float32), (384, 384), interpolation=cv2.INTER_CUBIC
        )
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float16)
        return img

    img = np.transpose(img, (2, 0, 1)).astype(np.float16)
    inputs_batched, intersections = disassemble(
        img,
        config["tile_res"],
        config["overlap"],
        batch_size=config["batch_size"],
        to_resize=config["to_resize"],
    )
    meta.intersection_map = intersections

    return inputs_batched, meta


# postprocessing logic for model
def unet_postprocessing(
    data: list | np.ndarray,
    meta: ImageMeta,
    config: dict,
    format_as: str,
    threshold: float | None,
    get_overlayed: bool = False,
):
    assert not (get_overlayed ^ (meta.src_img is not None)), (
        "Provide source image and flag"
    )

    if isinstance(data, list):
        if meta.intersection_map is None or len(meta.intersection_map.shape) != 2:
            raise ValueError("Intersection map mismatch")
        data = (
            assebmle_logit(
                data, (meta.height, meta.width), meta.tile_resolution, config["overlap"]
            )
            / meta.intersection_map
        )

    data = np.squeeze(data)
    probs = sigmoid(data)
    result = {"heatmap": handle_and_encode(probs * 255, format_as, is_mask=True)}

    if threshold:
        mask = np.where(probs >= threshold, 1, 0)
        if get_overlayed:
            overlayed = overlay_img_mask(meta.src_img, mask)
            result["overlayed"] = handle_and_encode(overlayed, format_as)
        result["mask"] = handle_and_encode(mask * 255, format_as, is_mask=True)
    elif get_overlayed:
        overlayed = overlay_img_mask(meta.src_img, probs)
        result["overlayed"] = handle_and_encode(overlayed, format_as)

    return result


# dicts to map current models with processing pipelines
PREPROCESSING_PIPELINE = {"unet_fp16": unet_preprocessing}
POSTPROCESSING_PIPELINE = {"unet_fp16": unet_postprocessing}
