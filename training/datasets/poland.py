from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from albumentations import ToTensorV2
from cv2 import imread, IMREAD_GRAYSCALE, cvtColor, COLOR_BGR2RGB


class PolandDataset(Dataset):
    images_poland_path = Path(r"data/landcover.ai.v1/output")
    all_imgs_paths = list(Path.rglob(images_poland_path, "*.jpg"))
    all_masks_paths = list(Path.rglob(images_poland_path, "*.png"))
    all_data_paths = list()

    for img in all_imgs_paths:
        filename = str(img).split("\\")[-1].removesuffix(".jpg")
        m_path = all_masks_paths.index(
            Path("data/landcover.ai.v1/output/" + filename + "_m.png")
        )
        all_data_paths.append((img, all_masks_paths[m_path]))
    all_data_paths = np.array(all_data_paths, dtype=tuple)

    def __init__(self, *, idx: np.array = None, transforms=ToTensorV2()):
        """Lazy loading and preparation of LandCover.ai dataset

        Args:
            idx (np.array, optional): Array of indexes for root directory of images. Defaults to None.
            transforms (_type_, optional): Augmentations. Defaults to ToTensorV2().

        Raises:
            ValueError: Index not specified.
        """
        if idx is None:
            raise ValueError("idx not specified!")

        self._data_paths = PolandDataset.all_data_paths[idx]
        self._transforms = transforms
        self._len = len(self._data_paths)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img_path, mask_path = self._data_paths[idx]
        img, msk = imread(img_path), imread(mask_path, IMREAD_GRAYSCALE)
        img = cvtColor(img, COLOR_BGR2RGB)
        for val in [2, 3, 4]:
            msk[msk == val] = 0

        if self._transforms is not None:
            augmented = self._transforms(image=img, mask=msk)
            img_tsr, msk_tsr = augmented["image"], augmented["mask"]
            return img_tsr, msk_tsr
        else:
            return img, msk
