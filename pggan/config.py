import json
import os

import torch


class _Config:
    N_CHANNEL = 1
    LATENT_VECTOR_SIZE = 256
    FEATURE_DIM_GENERATOR = 256
    FEATURE_DIM_DISCRIMINATOR = 256
    MIN_RESOLUTION = 2  # start from 2
    MAX_RESOLUTION = 6  # 2 ** 7 = 128
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.87
    BETA1 = 0.0
    BETA2 = 0.99

    TRANSITION_IMAGES_NUM = 200 * 500
    STABILIZATION_IMAGES_NUM = 100 * 500
    LEVEL_IMAGES_NUM = (TRANSITION_IMAGES_NUM + STABILIZATION_IMAGES_NUM) * 2
    DATA_LOADER_WORKERS = os.cpu_count()
    BATCH_SIZE = {r: 2 ** (11 - r) for r in range(2, 11)}
    SNAPSHOT_EPOCH_INTERVAL = 10

    DATASET = "kanji"

    N_LEVEL = MAX_RESOLUTION - MIN_RESOLUTION + 5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Config = _Config()

_skip_keys = ["DEVICE"]


def dump(path):
    d = {k: getattr(Config, k) for k in Config.__dir__() if not (k.startswith("_") or k in _skip_keys)}
    with open(path, "w") as fp:
        json.dump(d, fp)


def load(path):
    with open(path, "r") as fp:
        d = json.load(fp)
    for k, v in d.items():
        setattr(Config, k, v)
