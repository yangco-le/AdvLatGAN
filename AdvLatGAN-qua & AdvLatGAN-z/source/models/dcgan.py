import torch
import torch.nn as nn
import torch.nn.init as init
import math


class Generator(nn.Module):
    def __init__(self, z_dim, M):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 1024, M, 1, 0, bias=False),  # 4, 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z.view(-1, self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, M):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 64
            nn.Conv2d(3, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            # 16
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            # 8
            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
            # 4
        )

        self.linear = nn.Linear(M // 16 * M // 16 * 512, 1)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=2)


class Generator48(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=3)


class Generator64(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=4)


class Generator128(Generator):
    def __init__(self, z_dim):
        super().__init__(z_dim, M=8)


class Discriminator32(Discriminator):
    def __init__(self):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self):
        super().__init__(M=48)


class Discriminator64(Discriminator):
    def __init__(self):
        super().__init__(M=64)


class Discriminator128(Discriminator):
    def __init__(self):
        super().__init__(M=128)


class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResGenerator32(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)

        self.blocks = nn.Sequential(
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        res_arch_init(self)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 256, 4, 4)
        return self.blocks(inputs)


class ResGenerator48(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 6 * 6 * 512)

        self.blocks = nn.Sequential(
            ResGenBlock(512, 256),
            ResGenBlock(256, 128),
            ResGenBlock(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        res_arch_init(self)

    def forward(self, z):
        inputs = self.linear(z)
        inputs = inputs.view(-1, 512, 6, 6)
        return self.blocks(inputs)


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = nn.Linear(128, 1, bias=False)
        res_arch_init(self)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


class ResDiscriminator48(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 64),
            ResDisBlock(64, 128, down=True),
            ResDisBlock(128, 256, down=True),
            ResDisBlock(256, 512, down=True),
            nn.ReLU())
        self.linear = nn.Linear(512, 1, bias=False)
        res_arch_init(self)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
