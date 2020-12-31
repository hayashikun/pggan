import logging

from pggan.config import Config

logging.basicConfig(level=logging.INFO)


def _set_debug_config_idol():
    Config.N_CHANNEL = 3
    Config.DATASET = "idol"


def _set_debug_config_kamon():
    Config.N_CHANNEL = 1
    Config.DATASET = "kamon"


def _set_debug_config():
    Config.MAX_RESOLUTION = 6
    Config.N_LEVEL = Config.MAX_RESOLUTION - Config.MIN_RESOLUTION + 4
    Config.LATENT_VECTOR_SIZE = 64
    Config.FEATURE_DIM_GENERATOR = 256
    Config.FEATURE_DIM_DISCRIMINATOR = 256
    Config.TRANSITION_IMAGES_NUM = 200 * 4
    Config.STABILIZATION_IMAGES_NUM = 100 * 4
    Config.LEVEL_IMAGES_NUM = (Config.TRANSITION_IMAGES_NUM + Config.STABILIZATION_IMAGES_NUM) * 2
    Config.BATCH_SIZE = {r: 2 ** (10 - r) for r in range(2, 11)}
    Config.SNAPSHOT_EPOCH_INTERVAL = 1

    _set_debug_config_kamon()


def load_dataset(name=None):
    from pggan import dataset
    dataset.load_dataset(name)


def train(debug=False):
    from pggan.trainer import Trainer
    logging.info(f"Device: {Config.DEVICE}")
    if debug:
        logging.info("Debug mode")
        _set_debug_config()

    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    import fire

    fire.Fire({
        "train": train,
        "load_dataset": load_dataset
    })
