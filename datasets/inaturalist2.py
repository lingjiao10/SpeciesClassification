import os 
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torchvision.datasets.vision import VisionDataset

# from treelib import Tree, Node
# import pandas as pd



class INaturalist(VisionDataset):

    def __init__(self, root, version, transform: Optional[Callable] = None, insecta=False):
        super().__init__(root, transform=transform)

        # read json file
        annofile = os.path.join(root, "{}.json".format(version))
        with open(annofile, 'r') as file:
            data = json.load(file)
            categories = data["categories"]
            images = data["images"]
            annotations = data["annotations"]

        # map: category_id -> class_id
        self.categories_index: Dict[int, int] = {}
        class_id = 0
        for cat in categories:
            if(cat["class"] == "Insecta"):
                self.categories_index[cat["id"]] = class_id
                class_id += 1
        # print()

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, int, str]] = []
        assert(len(images)==len(annotations))
        for i in range(len(images)):
            image = images[i]
            anno = annotations[i]
            assert(image["id"]==anno["image_id"])
            category = categories[anno["category_id"]]
            if(insecta):
                if(category["class"] == "Insecta"):
                    self.index.append((image["id"], anno["category_id"], image["file_name"]))
            else:
                self.index.append((int(image["id"]), int(anno["category_id"]), image["file_name"]))
            

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        image_id, cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, "images", fname))

        # target: Any = []
        # for t in self.target_type:
        #     if t == "full":
        #         target.append(cat_id)
        #     else:
        #         target.append(self.categories_map[cat_id][t])
        # target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        target = self.categories_index[cat_id]

        return img, target


if __name__ == "__main__":
    nature = INaturalist(root="/home/h3c/wangcong/.cache/lavis/inaturalist/2021", version="val", insecta=True)
    print(len(nature))
    print(nature[100])