import os

import torch.utils.data as data_utils
from torchvision import datasets, transforms

import s3
from pggan import DatasetsDirectoryPath
from pggan.config import Config


def load_dataset():
    dataset_path = os.path.join(DatasetsDirectoryPath, Config.DATASET)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        s3.sync(dataset_path, f"datasets/{Config.DATASET}/")


def dataloader(resolution):
    batch_size = Config.BATCH_SIZE[resolution]
    image_size = 2 ** resolution

    dataset = datasets.ImageFolder(root=DatasetsDirectoryPath,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.ToTensor(),
                                   ]),
                                   is_valid_file=lambda x: f"{Config.DATASET}/" in x)
    dl = data_utils.DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=Config.DATA_LOADER_WORKERS)
    return dl
