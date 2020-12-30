import torch


class Config:
    N_CHANNEL = 3
    LATENT_VECTOR_SIZE = 512
    FEATURE_DIM_GENERATOR = 512
    FEATURE_DIM_DISCRIMINATOR = 512
    MIN_RESOLUTION = 2  # start from 2
    MAX_RESOLUTION = 8  # 2 ** 8 = 256
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.87
    BETA1 = 0.0
    BETA2 = 0.99

    TRANSITION_IMAGES_NUM = 200 * 500
    STABILIZATION_IMAGES_NUM = 100 * 500
    LEVEL_IMAGES_NUM = (TRANSITION_IMAGES_NUM + STABILIZATION_IMAGES_NUM) * 2
    DATA_LOADER_WORKERS = 4
    BATCH_SIZE = {r: 2 ** (13 - r) for r in range(2, 11)}

    DATASET = "CelebA-HQ"

    N_LEVEL = MAX_RESOLUTION - MIN_RESOLUTION + 5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
