import dataset
from pggan.config import Config
from pggan.trainer import Trainer


def _set_debug_config():
    Config.MAX_RESOLUTION = 6
    Config.LATENT_VECTOR_SIZE = 32
    Config.FEATURE_DIM_GENERATOR = 64
    Config.FEATURE_DIM_DISCRIMINATOR = 64
    Config.TRANSITION_IMAGES_NUM = 200 * 4
    Config.STABILIZATION_IMAGES_NUM = 100 * 4
    Config.LEVEL_IMAGES_NUM = (Config.TRANSITION_IMAGES_NUM + Config.STABILIZATION_IMAGES_NUM) * 2
    dataset._batch_size = {
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
