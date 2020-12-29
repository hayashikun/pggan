import os

import torch.utils.data as data_utils
from torchvision import datasets, transforms

from pggan import DataDirectoryPath
from pggan.config import Config

_batch_size = {
    2: 256,
    3: 128,
    4: 64,
    5: 32,
    6: 16,
    7: 8,
    8: 4,
    9: 2,
    10: 1,
}


def dataloader(resl):
    batch_size = _batch_size[resl]
    image_size = 2 ** resl

    celeb_data_root = os.path.join(DataDirectoryPath, "CelebAMask-HQ")

    dataset = datasets.ImageFolder(root=celeb_data_root,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.ToTensor(),
                                   ]),
                                   is_valid_file=lambda x: "CelebA-HQ-img/" in x)
    dl = data_utils.DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=Config.DATA_LOADER_WORKERS)
    return dl
