from functools import lru_cache
from itertools import permutations
from typing import Optional

import numpy as np
import torch
import torchvision
from avalanche.benchmarks import nc_benchmark

from avalanche.benchmarks.classic.ccifar10 import _default_cifar10_train_transform, _default_cifar10_eval_transform

from avalanche.benchmarks.classic.ccifar100 import _default_cifar100_train_transform, _default_cifar100_eval_transform

from avalanche.benchmarks.classic.ctiny_imagenet import _get_tiny_imagenet_dataset

from avalanche.benchmarks.classic.ctiny_imagenet import \
    _default_train_transform as _default_tiny_imagenet_train_transform
from avalanche.benchmarks.classic.ctiny_imagenet import \
    _default_eval_transform as _default_tiny_imagenet_eval_transform
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.external_datasets import get_cifar10_dataset, \
    get_cifar100_dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
from torchvision.transforms import transforms


from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from PIL import Image

# from .folder import find_classes, make_dataset
# from .utils import download_and_extract_archive, verify_str_arg
# from vision import VisionDataset


class Imagenette(VisionDataset):
    """`Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ image classification dataset.

    Args:
        root (string): Root directory of the Imagenette dataset.
        split (string, optional): The dataset split. Supports ``"train"`` (default), and ``"val"``.
        size (string, optional): The image size. Supports ``"full"`` (default), ``"320px"``, and ``"160px"``.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version, e.g. ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class name, class index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (WordNet ID, class index).
    """

    _ARCHIVES = {
        "full": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", "fe2fc210e6bb7c5664d602c3cd71e612"),
        "320px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "3df6f0d01a2c9592104656642f5e78a3"),
        "160px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "e793b78cc4c9e9a4ccc0c1155377a412"),
    }
    _WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chain saw", "chainsaw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        size: str = "full",
        download=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ["train", "val"])
        self._size = verify_str_arg(size, "size", ["full", "320px", "160px"])

        self._url, self._md5 = self._ARCHIVES[self._size]
        self._size_root = Path(self.root) / Path(self._url).stem
        self._image_root = str(self._size_root / self._split)

        if download:
            self._download()
        elif not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.wnids, self.wnid_to_idx = find_classes(self._image_root)
        self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            class_name: idx for wnid, idx in self.wnid_to_idx.items() for class_name in self._WNID_TO_CLASS[wnid]
        }
        self._samples = make_dataset(self._image_root, self.wnid_to_idx, extensions=".jpeg")
        self.targets = [y for x, y in self._samples]

    def _check_exists(self) -> bool:
        return self._size_root.exists()

    def _download(self):
        # if self._check_exists():
        #     raise RuntimeError(
        #         f"The directory {self._size_root} already exists. "
        #         f"If you want to re-download or re-extract the images, delete the directory."
        #     )
        if not self._check_exists():
            download_and_extract_archive(self._url, self.root, md5=self._md5)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)

def get_permutation(n_classes):
    @lru_cache(maxsize=None)
    def rule_asc(n):
        a = [0 for i in range(n + 1)]
        k = 1
        a[1] = n
        while k != 0:
            x = a[k - 1] + 1
            y = a[k] - 1
            k -= 1
            while x <= y:
                a[k] = x
                y -= x
                k += 1
            a[k] = x + y
            yield a[:k + 1]

    divisions = list(filter(lambda x: not 1 in x and len(x) > 1,
                            rule_asc(n_classes)))

    all_divisions = []
    for d in divisions:
        all_divisions.extend(set(permutations(d)))

    selected = all_divisions[np.random.randint(0, len((all_divisions)))]

    return {i: v for i, v in enumerate(selected)}


def get_dataset_by_name(name: str, root: str = None):
    name = name.lower()

    if name == 'cifar10':
        train_set, test_set = get_cifar10_dataset(root)
        train_t = _default_cifar10_train_transform
        test_t = _default_cifar10_eval_transform
    elif name == 'cifar100':
        train_set, test_set = get_cifar100_dataset(root)
        train_t = _default_cifar100_train_transform
        test_t = _default_cifar100_eval_transform
    elif name == 'tinyimagenet':
        train_set, test_set = _get_tiny_imagenet_dataset(root)
        train_t = _default_tiny_imagenet_train_transform
        train_t = transforms.Compose([train_t, transforms.RandomCrop(size=(64, 64), padding=8)])

        test_t = _default_tiny_imagenet_eval_transform
    elif name == 'imagenette':
        if root is None:
            root = default_dataset_location("imagenette")

        train_set = Imagenette(root=root, split='train', size='320px', download=True)
        test_set = Imagenette(root=root, split='train', size='320px')
        train_t = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=(320, 320), padding=32),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_t = transforms.Compose(
            [
                transforms.Resize(size=(320, 320)),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        # for d in [train_set, test_set]:
        #     ys = [y for x, y in d]
        #     d.targets = np.asarray(ys)
    else:
        return None

    return train_set, test_set, train_t, test_t



def get_dataset_nc_scenario(name: str, n_tasks: int, til: bool,
                            shuffle: bool = True, seed: Optional[int] = None,
                            force_sit=False, method_name=None, dev_split=None,
                            permuted_dataset: bool = False):
    name = name.lower()

    r = get_dataset_by_name(name)

    if r is None:
        assert False, f'Dataset {name} not found.'

    train_split, test_split, train_t, test_t = r

    if permuted_dataset:
        rp = get_permutation(len(set(train_split.targets)))
        n_tasks = len(rp)
    else:
        rp = None

    if dev_split is not None:
        idx = np.arange(len(train_split))
        np.random.RandomState(0).shuffle(idx)

        if isinstance(dev_split, int):
            dev_i = dev_split
        else:
            dev_i = int(len(idx) * dev_split)

        dev_idx = idx[:dev_i]
        train_idx = idx[dev_i:]

        test_split = torch.utils.data.Subset(train_split, dev_idx)
        train_split = torch.utils.data.Subset(train_split, train_idx)

    if method_name == 'cope':
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            fixed_class_order=None,
            per_exp_classes=rp,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=False,
            class_ids_from_zero_from_first_exp=True,
            train_transform=train_t,
            eval_transform=test_t)

    if til and not force_sit:
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=True,
            seed=seed,
            fixed_class_order=None,
            per_exp_classes=rp,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_t,
            eval_transform=test_t)
    else:
        return nc_benchmark(
            train_dataset=train_split,
            test_dataset=test_split,
            n_experiences=n_tasks,
            task_labels=False,
            seed=seed,
            class_ids_from_zero_from_first_exp=True,
            fixed_class_order=None,
            per_exp_classes=rp,
            shuffle=shuffle,
            train_transform=train_t,
            eval_transform=test_t)
