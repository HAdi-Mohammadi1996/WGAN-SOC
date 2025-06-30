
import torch
from torchsummary import summary
import torch.nn.utils.parametrizations as p

class SNDense(torch.nn.Module):
    def __init__(self, in_channels, out_channel, cube_size):
        super(SNDense, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.cube_size = cube_size
        self.fc1 = torch.nn.Linear(self.in_channels * self.cube_size**3, out_channel, bias=False)
        p.spectral_norm(self.fc1, 'weight')

    def forward(self, x):
        return self.fc1(x)

class SNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SNConv, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        p.spectral_norm(self.conv, 'weight')

    def forward(self, x):
        return self.conv(x)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, ndf, cube_size):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            # D1
            SNConv(in_channels, ndf, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            SNConv(ndf, ndf, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            # D2
            SNConv(ndf, ndf*2, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            SNConv(ndf*2, ndf*2, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            # D3
            SNConv(ndf*2, ndf*4, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            SNConv(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            # D4
            SNConv(ndf*4, ndf*8, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            SNConv(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            # D5
            SNConv(ndf*8, ndf*16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc1 = SNDense(ndf*16, 1, cube_size)   

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return self.fc1(x)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Discriminator(in_channels=1, out_channels=1, cube_size=4)
# model.to(device)
# summary(model, input_size=(1, 64, 64, 64), device=device.type)

