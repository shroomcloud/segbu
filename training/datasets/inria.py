from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from albumentations import ToTensorV2
from cv2 import imread, IMREAD_GRAYSCALE, cvtColor, COLOR_BGR2RGB
from datasets import disassemble


class InriaDataset(Dataset):
    images_inria_path = Path(r"data/AerialImageDataset/train/images")
    all_imgs_paths = list(Path.rglob(images_inria_path, "*.tif"))
    all_masks_paths = [
        Path(rf"data/AerialImageDataset/train/gt/{i.name}") for i in all_imgs_paths
    ]
    all_data_paths = np.fromiter(zip(all_imgs_paths, all_masks_paths), dtype=tuple)

    def __init__(
        self,
        *,
        mode: str = "train",
        idx: np.array = None,
        res: int = 512,
        overlap: float = 0.2,
        transforms=ToTensorV2(),
    ):
        """Preloads and prepares data of INRIA dataset

        Args:
            mode (str, optional): train / val / test. Defaults to "train".
            idx (np.array, optional): Array of indexes for root directory of images.
            res (int, optional): Tile resolution. Defaults to 512.
            overlap (float, optional): Overlap percent. Defaults to 0.15.
            transforms (_type_, optional): Augmentations. Defaults to ToTensorV2().

        Raises:
            ValueError: Wrong mode.
            ValueError: Index not specified.
        """
        if mode not in ("train", "val", "test"):
            print(f"{mode} is not correct")
            raise ValueError("Wrong mode")

        if idx is None:
            raise ValueError("idx not specified!")

        self._data_paths = InriaDataset.all_data_paths[idx]
        self._mode = mode
        self._res = res
        self._overlap = overlap
        self._transforms = transforms

        self._data = []
        for path in self._data_paths:
            self._data.extend(
                self._make_tile(path, res=self._res, overlap=self._overlap)
            )

        self._len = len(self._data)

    def __len__(self):
        return self._len

    @staticmethod
    def _make_tile(paths: tuple, res: int, overlap: float):
        img_path, mask_path = paths
        img, mask = imread(img_path), imread(mask_path, IMREAD_GRAYSCALE)
        img = cvtColor(img, COLOR_BGR2RGB)
        mask[mask == 255] = 1

        return disassemble.disassebmle_w_mask(img, mask, res, overlap)

    def __getitem__(self, idx):
        img, msk = self._data[idx]
        if self._transforms is not None:
            augmented = self._transforms(image=img, mask=msk)
            img_tsr, msk_tsr = augmented["image"], augmented["mask"]
            return img_tsr, msk_tsr
        else:
            return img, msk
