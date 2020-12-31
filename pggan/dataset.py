import os

import torch.utils.data as data_utils
from torchvision import datasets, transforms

from pggan import DatasetsDirectoryPath
from pggan import s3
from pggan.config import Config


def load_dataset(name=None):
    if name is None:
        name = Config.DATASET
    dataset_path = os.path.join(DatasetsDirectoryPath, name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    s3.sync(dataset_path, f"datasets/{name}/")


def dataloader(resolution):
    batch_size = Config.BATCH_SIZE[resolution]
    image_size = 2 ** resolution

    transform_components = list()
    if Config.N_CHANNEL == 1:
        transform_components.append(transforms.Grayscale())
    transform_components += [transforms.Resize(image_size), transforms.ToTensor(), ]
    if Config.N_CHANNEL == 3:
        transform_components.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    dataset = datasets.ImageFolder(root=DatasetsDirectoryPath,
                                   transform=transforms.Compose(transform_components),
                                   is_valid_file=lambda x: f"{Config.DATASET}/" in x)
    dl = data_utils.DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=Config.DATA_LOADER_WORKERS)
    return dl
