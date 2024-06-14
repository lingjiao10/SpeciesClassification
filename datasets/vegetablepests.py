import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math

from PIL import Image

import torch

from torchvision.datasets.vision import VisionDataset


class VegetablePests(VisionDataset):
    def __init__(
        self,
        root:str,
        class_name_txt: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        

    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        # self.classname_df = pd.read_csv(os.path.join(root,"classnames.txt"))
        self.classname_df = pd.read_csv(class_name_txt, sep='|')
        # self.classnames =
        # print(self.classname_df) 
        # print(self.classname_df.loc[0][1])

        os.makedirs(root, exist_ok=True)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")


        self.all_categories = sorted([di for di in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, di))])
        # print(self.all_categories)
        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.all_categories):
            files = os.listdir(os.path.join(self.root, dir_name))
            for fname in files:
                self.index.append((dir_index, fname))


    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        target: Any = []
        target.append(self.classname_df.loc[cat_id][0])
        # for t in self.classname_df.loc[cat_id][1:]:
        #     # print(t)
        #     # print(type(t))
        #     if isinstance(t, str):
        #         target.append(t)

        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

if __name__ == "__main__":
    # dataset = VegetablePests("~/wangcong/Datasets/vegetable_pests")
    root_dir = "/home/h3c/wangcong/.cache/lavis/vegetable_pests"
    dataset = VegetablePests(root=root_dir+'/images/train', 
            class_name_txt=root_dir+"/classnames.txt")

    # print(len(dataset))
    print(dataset[12])
