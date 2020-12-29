import unittest

from pggan import trainer
from pggan.config import Config

BATCH_SIZE = 7


class TrainerTestCase(unittest.TestCase):
    def setUp(self):
        self.trainer = trainer.Trainer()
        Config.TRANSITION_IMAGES_NUM = 200
        Config.STABILIZATION_IMAGES_NUM = 100
        Config.LEVEL_IMAGES_NUM = (Config.TRANSITION_IMAGES_NUM + Config.STABILIZATION_IMAGES_NUM) * 2
        Config.MAX_RESOLUTION = 6

    def fadein(self):
        g_fadein = getattr(self.trainer.generator.model, "fadein_module", None)
        d_fadein = getattr(self.trainer.discriminator.model, "fadein_module", None)
        return g_fadein, d_fadein

    def test_batch_trained(self):
        self.trainer.batch_trained(BATCH_SIZE)
        self.assertEqual(BATCH_SIZE, self.trainer.trained_image_number)
        self.assertEqual(2, self.trainer.resolution)
        gf, df = self.fadein()
        self.assertIsNone(gf)
        self.assertIsNone(df)
        for _ in range(Config.LEVEL_IMAGES_NUM // BATCH_SIZE - 1):
            self.trainer.batch_trained(BATCH_SIZE)
        self.assertEqual(2, self.trainer.resolution)

        # Resolution 2 -> 3, G transition
        self.trainer.batch_trained(BATCH_SIZE)
        self.assertEqual(3, self.trainer.resolution)
        gf, df = self.fadein()
        self.assertGreater(gf.alpha, 0)
        self.assertEqual(0, df.alpha)

        while self.trainer.trained_image_number % Config.LEVEL_IMAGES_NUM <= Config.TRANSITION_IMAGES_NUM:
            self.trainer.batch_trained(BATCH_SIZE)

        # G stabilization
        gf, df = self.fadein()
        self.assertEqual(1, gf.alpha)
        self.assertEqual(0, df.alpha)

        while self.trainer.trained_image_number % Config.LEVEL_IMAGES_NUM <= Config.LEVEL_IMAGES_NUM // 2:
            self.trainer.batch_trained(BATCH_SIZE)

        # D transition
        gf, df = self.fadein()
        self.assertIsNone(gf)
        self.assertGreater(df.alpha, 0)

        while self.trainer.trained_image_number % Config.LEVEL_IMAGES_NUM <= \
                Config.LEVEL_IMAGES_NUM // 2 + Config.TRANSITION_IMAGES_NUM:
            self.trainer.batch_trained(BATCH_SIZE)

        # D stabilization
        gf, df = self.fadein()
        self.assertIsNone(gf)
        self.assertEqual(1, df.alpha)

        while self.trainer.resolution < Config.MAX_RESOLUTION:
            self.trainer.batch_trained(BATCH_SIZE)

        self.assertEqual(Config.MAX_RESOLUTION, self.trainer.resolution)
        for _ in range(Config.LEVEL_IMAGES_NUM // BATCH_SIZE + 1):
            self.trainer.batch_trained(BATCH_SIZE)
        self.assertEqual(Config.MAX_RESOLUTION, self.trainer.resolution)
        gf, df = self.fadein()
        self.assertIsNone(gf)
        self.assertIsNone(df)


if __name__ == '__main__':
    unittest.main()
