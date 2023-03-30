import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

from treelib import Tree, Node
import pandas as pd
import numpy as np
from tqdm import tqdm

CATEGORIES_2021 = ["kingdom", "phylum", "class", "order", "family", "genus"]

DATASET_URLS = {
    "2017": "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz",
    "2018": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz",
    "2019": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz",
    "2021_train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz",
    "2021_train_mini": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz",
    "2021_valid": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
}

DATASET_MD5 = {
    "2017": "7c784ea5e424efaec655bd392f87301f",
    "2018": "b1c6952ce38f31868cc50ea72d066cc3",
    "2019": "c60a6e2962c9b8ccbd458d12c8582644",
    "2021_train": "e0526d53c7f7b2e3167b2b43bb2690ed",
    "2021_train_mini": "db6ed8330e634445efc8fec83ae81442",
    "2021_valid": "f6f6e0e242e3d4c9569ba56400938afc",
}


class INaturalist(VisionDataset):
    """`iNaturalist <https://github.com/visipedia/inat_comp>`_ Dataset.
    Args:
        root (string): Root directory of dataset where the image files are stored.
            This class does not require/use annotation files.
        version (string, optional): Which version of the dataset to download/use. One of
            '2017', '2018', '2019', '2021_train', '2021_train_mini', '2021_valid'.
            Default: `2021_train`.
        target_type (string or list, optional): Type of target to use, for 2021 versions, one of:
            - ``full``: the full category (species)
            - ``kingdom``: e.g. "Animalia"
            - ``phylum``: e.g. "Arthropoda"
            - ``class``: e.g. "Insecta"
            - ``order``: e.g. "Coleoptera"
            - ``family``: e.g. "Cleridae"
            - ``genus``: e.g. "Trichodes"
            for 2017-2019 versions, one of:
            - ``full``: the full (numeric) category
            - ``super``: the super category, e.g. "Amphibians"
            Can also be a list to output a tuple with all specified target types.
            Defaults to ``full``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        build_tree (bool, optional): If true, build category tree, only used for 2021 versions.
    """

    def __init__(
        self,
        root: str,
        version: str = "2021_train",
        target_type: Union[List[str], str] = "full",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        build_tree = False,
        load_weight = False,
        tree_depth = 7,
    ) -> None:
        self.version = verify_str_arg(version, "version", DATASET_URLS.keys())

        super().__init__(os.path.join(root, version), transform=transform, target_transform=target_transform)

        os.makedirs(root, exist_ok=True)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.all_categories: List[str] = []

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {}

        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        # tree: all categories/ tag->name identifer->index+name
        self.category_tree: Tree = None
        # pandas table -> help to build tree
        self.cat_table = None
        # distance between each category -> calculated by category tree
        self.full_distance_matrix = []
        

        if not isinstance(target_type, list):
            target_type = [target_type]
        if self.version[:4] == "2021":
            self.target_type = [verify_str_arg(t, "target_type", ("full", *CATEGORIES_2021)) for t in target_type]
            self._init_2021()
            if build_tree:
                print('---------------> Start building tree of all categories......')
                self._build_tree(tree_depth)
                print('---------------> Building tree of all categories Finished.')
                print('---------------> Start calculating distance of all leaves......')
                self._build_distance_matrix(tree_depth)
                print('---------------> Calculating distance of all leaves Finished.')
            if load_weight:
                print('---------------> loading distance matrix......')
                self.full_distance_matrix = np.loadtxt("./full_distance_matrix_4.csv", delimiter=',')
        else:
            self.target_type = [verify_str_arg(t, "target_type", ("full", "super")) for t in target_type]
            self._init_pre2021()

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.all_categories):
            files = os.listdir(os.path.join(self.root, dir_name))
            for fname in files:
                self.index.append((dir_index, fname))


    def _init_2021(self) -> None:
        """Initialize based on 2021 layout"""

        self.all_categories = sorted(os.listdir(self.root))

        # map: category type -> name of category -> index
        self.categories_index = {k: {} for k in CATEGORIES_2021}

        table = []
        for dir_index, dir_name in enumerate(self.all_categories):
            pieces = dir_name.split("_")
            
            if len(pieces) != 8:
                raise RuntimeError(f"Unexpected category name {dir_name}, wrong number of pieces")
            if pieces[0] != f"{dir_index:05d}":
                raise RuntimeError(f"Unexpected category id {pieces[0]}, expecting {dir_index:05d}")
            cat_map = {}
            table.append(pieces)
            for cat, name in zip(CATEGORIES_2021, pieces[1:7]):
                if name in self.categories_index[cat]:
                    cat_id = self.categories_index[cat][name]
                else:
                    cat_id = len(self.categories_index[cat])
                    self.categories_index[cat][name] = cat_id
                cat_map[cat] = cat_id
            self.categories_map.append(cat_map)

        if len(table) > 0:
            self.cat_table = pd.DataFrame(table, columns=['ID', *CATEGORIES_2021, 'speicies'])
            self.cat_table.set_index('ID')
            self.cat_table.to_csv('cat_table.csv', index=False)

    def _init_pre2021(self) -> None:
        """Initialize based on 2017-2019 layout"""

        # map: category type -> name of category -> index
        self.categories_index = {"super": {}}

        cat_index = 0
        super_categories = sorted(os.listdir(self.root))
        for sindex, scat in enumerate(super_categories):
            self.categories_index["super"][scat] = sindex
            subcategories = sorted(os.listdir(os.path.join(self.root, scat)))
            for subcat in subcategories:
                if self.version == "2017":
                    # this version does not use ids as directory names
                    subcat_i = cat_index
                    cat_index += 1
                else:
                    try:
                        subcat_i = int(subcat)
                    except ValueError:
                        raise RuntimeError(f"Unexpected non-numeric dir name: {subcat}")
                if subcat_i >= len(self.categories_map):
                    old_len = len(self.categories_map)
                    self.categories_map.extend([{}] * (subcat_i - old_len + 1))
                    self.all_categories.extend([""] * (subcat_i - old_len + 1))
                if self.categories_map[subcat_i]:
                    raise RuntimeError(f"Duplicate category {subcat}")
                self.categories_map[subcat_i] = {"super": sindex}
                self.all_categories[subcat_i] = os.path.join(scat, subcat)

        # validate the dictionary
        for cindex, c in enumerate(self.categories_map):
            if not c:
                raise RuntimeError(f"Missing category {cindex}")

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
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)

    def category_name(self, category_type: str, category_id: int) -> str:
        """
        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus" or "super"
            category_id(int): an index (class id) from this category
        Returns:
            the name of the category
        """
        if category_type == "full":
            return self.all_categories[category_id]
        else:
            if category_type not in self.categories_index:
                raise ValueError(f"Invalid category type '{category_type}'")
            else:
                for name, id in self.categories_index[category_type].items():
                    if id == category_id:
                        return name
                raise ValueError(f"Invalid category id {category_id} for {category_type}")

    def _check_integrity(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def download(self) -> None:
        if self._check_integrity():
            raise RuntimeError(
                f"The directory {self.root} already exists. "
                f"If you want to re-download or re-extract the images, delete the directory."
            )

        base_root = os.path.dirname(self.root)

        download_and_extract_archive(
            DATASET_URLS[self.version], base_root, filename=f"{self.version}.tgz", md5=DATASET_MD5[self.version]
        )

        orig_dir_name = os.path.join(base_root, os.path.basename(DATASET_URLS[self.version]).rstrip(".tar.gz"))
        if not os.path.exists(orig_dir_name):
            raise RuntimeError(f"Unable to find downloaded files at {orig_dir_name}")
        os.rename(orig_dir_name, self.root)
        print(f"Dataset version '{self.version}' has been downloaded and prepared for use")

    def _build_tree(self, stop_idx) -> None:
        self.category_tree = Tree()
        rootnode = Node(tag=0, identifier=0)  #root node
        self.category_tree.add_node(rootnode)
        self._addchildren(rootnode, 0, stop_idx=stop_idx)

        print("depth of tree: ", self.category_tree.depth())

    def _addchildren(self, parent, category_type_index, grandparent_tag=None, stop_idx=7):
        """
        递归添加节点
        # kingdom/species的identifer：id_name
        # 其余的：id_parentname_name
        """
        # if category_type_index == len(CATEGORIES_2021)+1:
        if category_type_index == stop_idx:
            return #stop recursion


        if category_type_index == 0: 
            category_type = CATEGORIES_2021[category_type_index] #current children source
            # print('current children source: ', category_type)
            for name, id in self.categories_index[category_type].items():
                node = Node(tag=name, identifier=str(id) + '_' +name)
                self.category_tree.add_node(node, parent=parent)
                # print("add node -> ", node.identifier)
                self._addchildren(node, category_type_index + 1, stop_idx=stop_idx)
        else:
            #filter children by parent name
            if category_type_index == len(CATEGORIES_2021):
                children_cat_type = 'speicies'
            else:
                children_cat_type = CATEGORIES_2021[category_type_index] #current children source
            parent_cat_type = CATEGORIES_2021[category_type_index-1]

            #search children add grandparent's filter
            if category_type_index == len(CATEGORIES_2021): # as genus maybe dumplicated in different kingdom
                grandparent_cat_type = CATEGORIES_2021[category_type_index-2]
                children = self.cat_table.loc[
                (self.cat_table[parent_cat_type] == parent.tag) & (self.cat_table[grandparent_cat_type] == grandparent_tag), children_cat_type
                ].drop_duplicates()
            else:
                children = self.cat_table.loc[
                self.cat_table[parent_cat_type] == parent.tag, children_cat_type
                ].drop_duplicates()

            # print(children)
            for label, child in children.items():
                name = child
                if category_type_index == len(CATEGORIES_2021):
                    child_id = label
                else:
                    child_id = str(self.categories_index[children_cat_type][name]) + "_" + parent.tag

                node = Node(tag=name, identifier=str(child_id) + '_' +name)
                try:
                    self.category_tree.add_node(node, parent=parent)
                except:
                    # print(e)
                    print("error -> ", node.identifier, parent.identifier)
                # print("add node -> ", node.identifier)

                # if category_type_index == len(CATEGORIES_2021):
                #     continue #stop recursion
                # else:
                self._addchildren(node, category_type_index + 1, parent.tag, stop_idx)
               
        

    def _build_distance_matrix(self, tree_depth) -> None:
        #共同父节点到两个节点的深度之和

        leaves = self.category_tree.leaves()
        num_of_nodes = len(leaves)
        # num_of_nodes = 10       
        nodes = leaves
        # elif args.category_type in CATEGORIES_2021[:5]:
        #     num_of_nodes = len(self.categories_index[args.category_type])
        # else: # 去掉genus，动、植物有重名
        #     return
        
        self.full_distance_matrix = np.zeros([num_of_nodes, num_of_nodes])
        file = open("full_distance_matrix_{}.csv".format(tree_depth), "w")

        for i in tqdm(range(num_of_nodes)):
            node1 = nodes[i]
            # node1_parent = self.category_tree.parent(node1.identifier)
            for j in range(i+1, num_of_nodes):
                node2 = nodes[j]
                _parent = self.category_tree.parent(node2.identifier)

                # if i==j:
                #     continue
                while(_parent.is_root is not True):
                    if self.category_tree.is_ancestor(_parent.identifier, node1.identifier):
                        m = int(node1.identifier.split('_')[0])
                        n = int(node2.identifier.split('_')[0])

                        self.full_distance_matrix[m][n] = self.category_tree.depth(node1) + self.category_tree.depth(node2) - 2*self.category_tree.depth(_parent)
                        self.full_distance_matrix[n][m] = self.full_distance_matrix[m][n] 
                        break
                    else:
                        _parent = self.category_tree.parent(_parent.identifier)


            
        for i in range(num_of_nodes):
            file.write(",".join(str(d) for d in self.full_distance_matrix[i,:]))
            file.write("\n")
                
        file.close()
        # np.savetxt('full_distance_matrix--.csv', self.full_distance_matrix, delimiter=",")
        # pd.
        



if __name__ == "__main__":
    nature = INaturalist("h:/Datasets/iNaturalist/", version='2021_valid',target_type="genus", build_tree=True,
        tree_depth=4)
    print(len(nature))
    print(nature[0])
    # print(nature.all_categories)
    print(nature.category_tree.depth())