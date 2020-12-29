import fire

from pggan.trainer import Trainer


def train():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    fire.Fire({
        "train": train
    })
