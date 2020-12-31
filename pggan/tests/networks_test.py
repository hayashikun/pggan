import unittest

import torch

from pggan import networks
from pggan.config import Config

INPUT_VECTOR_SIZE = 17


class GeneratorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        Config.LATENT_VECTOR_SIZE = 32
        Config.MAX_RESOLUTION = 6
        self.generator = networks.Generator()
        self.fixed_noise = torch.randn(INPUT_VECTOR_SIZE, Config.LATENT_VECTOR_SIZE, 1, 1)

    def test_init_model(self):
        outputs = self.generator(self.fixed_noise)
        self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, Config.N_CHANNEL, 4, 4])
        self.assertIsNone(getattr(self.generator.model, "fadein_module", None))

    def test_grow_flush(self):
        for res in range(3, Config.MAX_RESOLUTION + 1):
            self.generator.grow()
            outputs = self.generator(self.fixed_noise)
            self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, Config.N_CHANNEL, 2 ** res, 2 ** res])
            self.assertIsNotNone(getattr(self.generator.model, "fadein_module", None))
            self.generator.flush()
            outputs = self.generator(self.fixed_noise)
            self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, Config.N_CHANNEL, 2 ** res, 2 ** res])
            self.assertIsNone(getattr(self.generator.model, "fadein_module", None))

    def test_skip(self):
        self.generator.skip(4)
        outputs = self.generator(self.fixed_noise)
        self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, Config.N_CHANNEL, 2 ** 4, 2 ** 4])


class DiscriminatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        Config.LATENT_VECTOR_SIZE = 32
        self.discriminator = networks.Discriminator()

    def test_init_model(self):
        img = torch.ones(INPUT_VECTOR_SIZE, Config.N_CHANNEL, 4, 4)
        outputs = self.discriminator(img)
        self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, 1])
        self.assertIsNone(getattr(self.discriminator.model, "fadein_module", None))

    def test_grow_flush(self):
        for res in range(3, Config.MAX_RESOLUTION + 1):
            img = torch.ones(INPUT_VECTOR_SIZE, Config.N_CHANNEL, 2 ** res, 2 ** res)
            self.discriminator.grow()
            outputs = self.discriminator(img)
            self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, 1])
            self.assertIsNotNone(getattr(self.discriminator.model, "fadein_module", None))
            self.discriminator.flush()
            outputs = self.discriminator(img)
            self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, 1])
            self.assertIsNone(getattr(self.discriminator.model, "fadein_module", None))

    def test_skip(self):
        self.discriminator.skip(4)
        img = torch.ones(INPUT_VECTOR_SIZE, Config.N_CHANNEL, 2 ** 4, 2 ** 4)
        outputs = self.discriminator(img)
        self.assertListEqual(list(outputs.size()), [INPUT_VECTOR_SIZE, 1])


if __name__ == '__main__':
    unittest.main()
