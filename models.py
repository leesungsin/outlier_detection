

import torch
import torch.nn as nn
import torch.nn.parallel

class AE(nn.Module):
    def __init__(self, num_in_channels, z_size=100, num_filters=32, ngpu=1):
        super().__init__()
        self.encoder = nn.Sequential(
            # expected input: 28x28
            nn.Conv2d(num_in_channels, num_filters, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2), # reduce input size 28x28 to 14x14, MNIST 는 object variant 하지 않으니, 이거보다 stride 를 2 로 하는게 나으려나?

            nn.Conv2d(num_filters, 2 * num_filters, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)
            # reduce input size 14x14 to 7*7
        )
        # apply conv2d with stride 2
        self.latent_space = nn.Conv2d(2*num_filters, z_size, 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, 2*num_filters, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(2*num_filters),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(2 * num_filters, num_filters, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(num_filters, num_in_channels, kernel_size=2, stride=1, padding=1),
            nn.Tanh() # -1 ~ 1
        )
        self.weight_init()

    def weight_init(self):
        self.encoder.apply(weight_init)
        self.latent_space.apply(weight_init)
        self.decoder.apply(weight_init)

    def forawrd(self, x):
        z = self.encoder(x)
        z = self.latent_space(z)
        decode_z = self.decoder(z)

        return decode_z, z


# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)