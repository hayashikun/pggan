from pggan.config import Config
from pggan.trainer import Trainer


def _set_debug_config():
    Config.MAX_RESOLUTION = 6
    Config.N_LEVEL = Config.MAX_RESOLUTION - Config.MIN_RESOLUTION + 4
    Config.LATENT_VECTOR_SIZE = 64
    Config.FEATURE_DIM_GENERATOR = 256
    Config.FEATURE_DIM_DISCRIMINATOR = 256
    Config.TRANSITION_IMAGES_NUM = 200 * 8
    Config.STABILIZATION_IMAGES_NUM = 100 * 8
    Config.LEVEL_IMAGES_NUM = (Config.TRANSITION_IMAGES_NUM + Config.STABILIZATION_IMAGES_NUM) * 2
    Config.BATCH_SIZE = {r: 2 ** (10 - r) for r in range(2, 11)}


def train(debug=False):
    if debug:
        _set_debug_config()

    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    import fire

    fire.Fire({
        "train": train
    })
