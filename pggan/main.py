import logging
import os

import torch

from pggan import SnapshotDirectoryPath
from pggan import dataset
from pggan.config import Config, load
from pggan.networks import Generator
from pggan.trainer import Trainer

logging.basicConfig(level=logging.INFO)


def _set_debug_config_idol():
    Config.N_CHANNEL = 3
    Config.DATASET = "idol"


def _set_debug_config_kamon():
    Config.N_CHANNEL = 1
    Config.DATASET = "kamon"


def _set_debug_config():
    Config.MAX_RESOLUTION = 6
    Config.N_LEVEL = Config.MAX_RESOLUTION - Config.MIN_RESOLUTION + 2
    Config.LATENT_VECTOR_SIZE = 64
    Config.FEATURE_DIM_GENERATOR = 256
    Config.FEATURE_DIM_DISCRIMINATOR = 256
    Config.TRANSITION_IMAGES_NUM = 200 * 8
    Config.STABILIZATION_IMAGES_NUM = 100 * 8
    Config.LEVEL_IMAGES_NUM = (Config.TRANSITION_IMAGES_NUM + Config.STABILIZATION_IMAGES_NUM) * 2
    Config.BATCH_SIZE = {r: 2 ** (10 - r) for r in range(2, 11)}
    Config.SNAPSHOT_EPOCH_INTERVAL = 1

    _set_debug_config_idol()


def load_dataset(name=None):
    dataset.load_dataset(name)


def train(debug=False):
    logging.info(f"Device: {Config.DEVICE}")
    if debug:
        logging.info("Debug mode")
        _set_debug_config()

    trainer = Trainer()
    trainer.train()


def convert_onnx(snapshot):
    load(os.path.join(SnapshotDirectoryPath, snapshot, "config.json"))
    generator = Generator()
    generator.skip(Config.MAX_RESOLUTION)
    state_dict = torch.load(os.path.join(SnapshotDirectoryPath, snapshot, "generator.pt"),
                            map_location=torch.device("cpu"))
    generator.load_state_dict(state_dict)
    x = torch.zeros(1, Config.LATENT_VECTOR_SIZE, 1, 1)
    torch.onnx.export(generator, x, os.path.join(SnapshotDirectoryPath, snapshot, "generator.onnx"),
                      opset_version=11, export_params=True)


if __name__ == '__main__':
    import fire

    fire.Fire({
        "train": train,
        "load_dataset": load_dataset,
        "convert_onnx": convert_onnx
    })
