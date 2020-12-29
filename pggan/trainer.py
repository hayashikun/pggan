import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import utils as vutils

from pggan import dataset, SnapshotDirectoryPath
from pggan.config import *
from pggan.networks import Generator, Discriminator


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = MIN_RESOLUTION
        self.level = 0

        self.dataloader = None
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.opt_g = None
        self.opt_d = None

        self.trained_image_number = 0
        self.lr = LEARNING_RATE
        self.snapshot_noise = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=self.device)

        self.level_updated()

    def level_updated(self):
        self.dataloader = dataset.dataloader(self.resolution)
        self.lr = LEARNING_RATE * LEARNING_RATE_DECAY ** (self.resolution - MIN_RESOLUTION)
        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(BETA1, BETA2))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(BETA1, BETA2))

    def batch_trained(self, batch_size):
        self.trained_image_number += batch_size
        trained_in_level = self.trained_image_number % LEVEL_IMAGES_NUM

        prev_level = self.level
        self.level = self.trained_image_number // LEVEL_IMAGES_NUM
        self.resolution = min(MAX_RESOLUTION, MIN_RESOLUTION + self.level)

        # fadein is not None only when model is grown and not flushed.
        g_fadein = getattr(self.generator.model, "fadein_module", None)
        d_fadein = getattr(self.discriminator.model, "fadein_module", None)

        if trained_in_level < LEVEL_IMAGES_NUM // 2:
            if prev_level < self.level:
                if d_fadein is not None:  # flush is not needed for MIN_RESOLUTION
                    self.discriminator.flush()
                if self.resolution < MAX_RESOLUTION:  # models are grown, when self.resolution < MAX_RESOLUTION
                    self.generator.grow()
                    g_fadein = self.generator.model.fadein_module
                    self.discriminator.grow()
            if g_fadein is not None:
                g_fadein.update_alpha(trained_in_level / TRANSITION_IMAGES_NUM)
        elif LEVEL_IMAGES_NUM // 2 < trained_in_level:
            if g_fadein is not None:
                self.generator.flush()
            if d_fadein is not None:
                d_fadein.update_alpha((trained_in_level - LEVEL_IMAGES_NUM // 2) / TRANSITION_IMAGES_NUM)

        # level incremented?
        return trained_in_level < batch_size

    def train(self):
        criterion = nn.MSELoss()
        self.trained_image_number = 0

        epoch = 0
        g_losses = list()
        d_losses = list()

        while self.trained_image_number // LEVEL_IMAGES_NUM < N_LEVEL:
            new_level = False
            d_loss_sum = 0
            g_loss_sum = 0

            for i, (images, _) in enumerate(self.dataloader, 0):
                images = images.to(self.device)
                batch_size = images.size(0)
                real_labels = torch.ones(batch_size, ).to(self.device)
                fake_labels = torch.zeros(batch_size, ).to(self.device)

                self.generator.zero_grad()
                self.discriminator.zero_grad()

                outputs = self.discriminator(images).view(-1)
                d_loss_real = criterion(outputs, real_labels)
                noise = torch.randn(batch_size, LATENT_VECTOR_SIZE, 1, 1, device=self.device)
                fake_images = self.generator(noise)
                outputs = self.discriminator(fake_images.detach()).view(-1)
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.opt_d.step()

                # Generator
                outputs = self.discriminator(fake_images).view(-1)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                self.opt_g.step()

                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()

                if self.batch_trained(batch_size):
                    new_level = True
                    break

            epoch += 1
            g_loss = g_loss_sum / len(self.dataloader)
            d_loss = d_loss_sum / len(self.dataloader)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            print(f"Ep: {epoch} - Lv: {self.level}/{N_LEVEL}\t| G Loss: {g_loss:.3f}, D Loss: {d_loss:.3f}")

            with torch.no_grad():
                snapshot_images = self.generator(self.snapshot_noise).detach().cpu()
            img = vutils.make_grid(snapshot_images, nrow=8, padding=2, normalize=True)
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.imshow(np.transpose(img, (1, 2, 0)))
            fig.tight_layout()
            fig.savefig(os.path.join(SnapshotDirectoryPath, f"gen_{epoch}.png"),
                        bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()

            if new_level:
                # plot loss
                fig, ax = plt.subplots()
                ax.plot(g_losses, label="Generator loss")
                ax.plot(d_losses, label="Discriminator loss")
                ax.legend()
                ax.set(xlabel="Epoch", ylabel="Loss")
                fig.tight_layout()
                fig.savefig(os.path.join(SnapshotDirectoryPath, f"loss.png"), dpi=300)
                plt.close()

                self.level_updated()

        # save model
        torch.save(self.generator.state_dict(), f"generator.pt")
        torch.save(self.discriminator.state_dict(), f"discriminator.pt")
