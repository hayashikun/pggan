import os

import torch.utils.data as data_utils
from torchvision import datasets, transforms

from pggan import DataDirectoryPath
from pggan.config import Config


def dataloader(resolution):
    batch_size = Config.BATCH_SIZE[resolution]
    image_size = 2 ** resolution

    data_root = os.path.join(DataDirectoryPath, "dataset")

    dataset = datasets.ImageFolder(root=data_root,
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
