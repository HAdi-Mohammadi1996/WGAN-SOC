
import torch
from torchsummary import summary

class LatentSpaceReshape(torch.nn.Module):
    def __init__(self, noise_dim, n_channels, cube_size):
        super(LatentSpaceReshape, self).__init__()
        self.n_channels = n_channels
        self.cube_size = cube_size
        self.fc1 = torch.nn.Linear(noise_dim, self.n_channels * cube_size**3, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x.view(-1, self.n_channels, self.cube_size, self.cube_size, self.cube_size)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(out_channels, momentum=0.8),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm3d(out_channels, momentum=0.8)
        )

    def forward(self, x):
        return self.conv(x)

class G(torch.nn.Module):
    def __init__(self, in_channel):
        super(G, self).__init__()
        self.up = torch.nn.ConvTranspose3d(in_channel, in_channel//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.double_conv = DoubleConv(in_channel//2, in_channel//2)

    def forward(self, x):
        x = self.up(x)
        return self.double_conv(x)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, in_channels, out_channels, cube_size):
        super(Generator, self).__init__()
        assert in_channels % 16 == 0, "in_channels must be divisible by 16"
        self.latent_space_reshape = LatentSpaceReshape(noise_dim, in_channels, cube_size)
        self.g1 = G(in_channels)
        self.g2 = G(in_channels//2)
        self.g3 = G(in_channels//4)
        self.g4 = G(in_channels//8)
        self.final_conv = torch.nn.Conv3d(in_channels//16, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.latent_space_reshape(x)
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.g4(x)
        x = self.final_conv(x)
        return torch.nn.functional.tanh(x)   # Use tanh if you want to normalize output to [-1, 1]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Generator(noise_dim=100, in_channels=256, out_channels=1, cube_size=4)
# model.to(device)
# summary(model, input_size=(1, 100), device=device.type)
