from pathlib import Path
import torch 
import torchvision.transforms as T
from typing import Any, Tuple
import pandas as pd
from argparse import Namespace
import cv2
import numpy as np

def hflip_bbox(bbox, W):
    """Flip bounding boxes horizontally.
    Args:
        bbox (~numpy.ndarray): See the table below.
        W: width of the image
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.
    """
    bbox = bbox.copy()
    x_max = W - bbox[:, 1]
    x_min = W - bbox[:, 3]
    bbox[:, 1] = x_min
    bbox[:, 3] = x_max
    return bbox 

class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, csv: str, 
    image_dir: str, preprocess: Any, size: tuple):
        self.image_folder = Path(image_dir)
        self.preprocess = preprocess
        self.data = pd.read_csv(csv)
        self.size = size

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_folder / self.data.iloc[idx]['image']
        age = torch.tensor([self.data.loc[idx, 'age']], dtype=torch.float32)
        
        try:
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, self.size)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(img_path)
            raise e
       
        return img, age 

    def flip_image(self, img, bboxes):
        width = img.shape[1]
        flip_bboxes = hflip_bbox(np.array(bboxes), width)
        flip_img = np.fliplr(img)
        return flip_img, flip_bboxes

class AgeDataModule:
    def __init__(self, dataset_hparams: Namespace):
        self.hparams = dataset_hparams
        
        self.train_dataset = AgeDataset(self.hparams.train_path, 
        self.hparams.image_folder, self.hparams.preprocess, self.hparams.size)
        self.test_dataset = AgeDataset(self.hparams.test_path, 
        self.hparams.image_folder, self.hparams.preprocess, self.hparams.size)
        self.val_dataset = AgeDataset(self.hparams.val_path, 
        self.hparams.image_folder, self.hparams.preprocess, self.hparams.size)

    def _balance_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        target = self.train_dataset.data['age']
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. /class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in target])).double()

        return torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        sampler = self._balance_sampler() if self.hparams.balance else None
        shuffle = not self.hparams.balance
        return torch.utils.data.DataLoader(self.train_dataset, 
        batch_size=self.hparams.batch_size,
        num_workers=self.hparams.num_workers,
        sampler=sampler,
        shuffle=shuffle)


    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_dataset, 
        batch_size=self.hparams.batch_size,
        num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, 
        batch_size=self.hparams.batch_size, 
        num_workers=self.hparams.num_workers)


