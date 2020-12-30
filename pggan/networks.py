import copy

import torch
from torch import nn

from pggan.config import Config


def copy_module(model, target=None, contain=True, to_model=None):
    if to_model is None:
        to_model = nn.Sequential()
    for name, module in model.named_children():
        if (contain and name == target) or (not contain and not name == target):
            to_model.add_module(name, module)
            to_model[-1].load_state_dict(module.state_dict())
    return to_model


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps) ** 0.5


class EqualizedConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, a=nn.init.calculate_gain("conv2d"))
        self.bias = torch.nn.Parameter(torch.zeros(ch_out, device=Config.DEVICE))
        self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)


class EqualizedLinear(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(EqualizedLinear, self).__init__()
        self.linear = nn.Linear(ch_in, ch_out, bias=False)
        nn.init.kaiming_normal_(self.linear.weight, a=nn.init.calculate_gain("linear"))
        self.bias = torch.nn.Parameter(torch.zeros(ch_out, device=Config.DEVICE))
        self.scale = (torch.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data / self.scale)

    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1, -1).expand_as(x)


class Fadein(nn.Module):
    def __init__(self, layer1, layer2):
        super(Fadein, self).__init__()
        self.alpha = 0.0
        self.layers = [layer1, layer2]

    def update_alpha(self, alpha):
        self.alpha = max(0, min(alpha, 1.0))

    def forward(self, x):
        return torch.add(self.layers[0](x).mul(1.0 - self.alpha), self.layers[1](x).mul(self.alpha))


# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class MinibatchStatConcatLayer(nn.Module):
    def __init__(self):
        super(MinibatchStatConcatLayer, self).__init__()
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(
            torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8
        )

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        target_shape[1] = 1
        vals = torch.mean(vals, dim=1, keepdim=True)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)


class DeConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding=0):
        super(DeConv, self).__init__()
        self.main = nn.Sequential(
            EqualizedConv2d(ch_in, ch_out, kernel_size, stride, padding),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.resolution = Config.MIN_RESOLUTION
        self.model = self.init_model()

    @staticmethod
    def first_module():
        ndim = Config.FEATURE_DIM_GENERATOR
        layers = [
            DeConv(Config.LATENT_VECTOR_SIZE, ndim, 4, 1, 3),
            DeConv(ndim, ndim, 3, 1, 1)
        ]
        return nn.Sequential(*layers), ndim

    def intermediate_module(self):
        halving = False
        ndim = Config.FEATURE_DIM_GENERATOR
        if self.resolution >= 6:
            halving = True
            ndim = ndim // (2 ** (self.resolution - 5))

        layers = [nn.Upsample(scale_factor=2, mode="nearest")]
        if halving:
            layers.append(DeConv(ndim * 2, ndim, 3, 1, 1))
        else:
            layers.append(DeConv(ndim, ndim, 3, 1, 1))
        layers.append(DeConv(ndim, ndim, 3, 1, 1))

        return nn.Sequential(*layers), ndim

    @staticmethod
    def to_rgb_module(ndim):
        return nn.Sequential(EqualizedConv2d(ndim, Config.N_CHANNEL, 1, 1, 0))

    def init_model(self):
        model = nn.Sequential()
        module, ndim = self.first_module()
        model.add_module("first_module", module)
        model.add_module("to_rgb_module", self.to_rgb_module(ndim))
        return model

    def grow(self):
        self.resolution += 1
        new_model = copy_module(self.model, "to_rgb_module", contain=False)

        prev_module = nn.Sequential()
        low_resl_to_rgb = copy_module(self.model, "to_rgb_module")
        prev_module.add_module("low_resl_upsample", nn.Upsample(scale_factor=2, mode="nearest"))
        prev_module.add_module("low_resl_to_rgb_module", low_resl_to_rgb)

        next_module = nn.Sequential()
        inter_module, ndim = self.intermediate_module()
        next_module.add_module("high_resl_module", inter_module)
        next_module.add_module("high_resl_to_rgb_module", self.to_rgb_module(ndim))

        new_model.add_module("fadein_module", Fadein(prev_module, next_module))
        self.model = new_model

    def flush(self):
        high_resl_module = copy_module(self.model.fadein_module.layers[1], "high_resl_module")
        high_resl_to_rgb = copy_module(self.model.fadein_module.layers[1], "high_resl_to_rgb_module")

        new_model = copy_module(self.model, "fadein_module", contain=False)
        new_model.add_module(f"intermediate_{self.resolution}", high_resl_module)
        new_model.add_module('to_rgb_module', high_resl_to_rgb)
        self.model = new_model

    def forward(self, x):
        return self.model(x.view(x.size(0), -1, 1, 1))


class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        self.main = nn.Sequential(
            EqualizedConv2d(ch_in, ch_out, kernel_size, stride, padding),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.resolution = Config.MIN_RESOLUTION
        self.model = self.init_model()

    @staticmethod
    def last_module():
        ndim = Config.FEATURE_DIM_DISCRIMINATOR
        layers = [
            MinibatchStatConcatLayer(),
            Conv(ndim + 1, ndim, 3, 1, 1),
            Conv(ndim, ndim, 4, 1, 0),
            Flatten(),
            EqualizedLinear(ndim, 1)
        ]
        return nn.Sequential(*layers), ndim

    def intermediate_module(self):
        halving = False
        ndim = Config.FEATURE_DIM_DISCRIMINATOR
        if self.resolution >= 6:
            halving = True
            ndim = ndim // (2 ** (self.resolution - 5))

        layers = [Conv(ndim, ndim, 3, 1, 1)]
        if halving:
            layers.append(Conv(ndim, ndim * 2, 3, 1, 1))
        else:
            layers.append(Conv(ndim, ndim, 3, 1, 1))
        layers.append(nn.AvgPool2d(kernel_size=2))

        return nn.Sequential(*layers), ndim

    @staticmethod
    def from_rgb_module(ndim):
        return nn.Sequential(Conv(Config.N_CHANNEL, ndim, 1, 1, 0))

    def init_model(self):
        model = nn.Sequential()
        module, ndim = self.last_module()
        model.add_module("from_rgb_module", self.from_rgb_module(ndim))
        model.add_module("last_module", module)
        return model

    def grow(self):
        self.resolution += 1

        prev_module = nn.Sequential()
        low_resl_from_rgb = copy_module(self.model, "from_rgb_module")
        prev_module.add_module("low_resl_downsample", nn.AvgPool2d(kernel_size=2))
        prev_module.add_module("low_resl_from_rgb_module", low_resl_from_rgb)

        next_module = nn.Sequential()
        inter_module, ndim = self.intermediate_module()
        next_module.add_module("high_resl_from_rgb_module", self.from_rgb_module(ndim))
        next_module.add_module("high_resl_module", inter_module)

        new_model = nn.Sequential()
        new_model.add_module("fadein_module", Fadein(prev_module, next_module))
        new_model = copy_module(self.model, "from_rgb_module", contain=False, to_model=new_model)
        self.model = new_model

    def flush(self):
        high_resl_module = copy_module(self.model.fadein_module.layers[1], "high_resl_module")
        high_resl_from_rgb = copy_module(self.model.fadein_module.layers[1], "high_resl_from_rgb_module")

        new_model = nn.Sequential()
        new_model.add_module('from_rgb_module', high_resl_from_rgb)
        new_model.add_module(f"intermediate_{self.resolution}", high_resl_module)

        new_model = copy_module(self.model, "fadein_module", contain=False, to_model=new_model)
        self.model = new_model

    def forward(self, x):
        return self.model(x)
