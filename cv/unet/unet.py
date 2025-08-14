import torch
from torch import nn

def make_conv_layer(
    in_channels: int, 
    out_channels: int, 
    kernel_size: tuple[int, int],
    padding: int = 0,
) -> nn.Sequential:
    layer = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            padding=padding
        ), 
        nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3),
            padding=padding
        ),
        nn.ReLU(inplace=True)
    )

    return layer

def center_crop(x: torch.Tensor, target: torch.Tensor):
    _, _, h1, w1 = x.shape
    _, _, h2, w2 = target.shape

    top = (h1 - h2) // 2
    left = (w1 - w2) // 2

    return x[:, :, top: top+h2, left: left+w2]

class UNet(nn.Module):
    """Full UNet implementation in PyTorch."""

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        self.down_layers = nn.ModuleList([
            make_conv_layer(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=0),
            make_conv_layer(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0),
            make_conv_layer(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0),
            make_conv_layer(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0),
            make_conv_layer(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=0),
        ])

        self.up_layers = nn.ModuleList([
            make_conv_layer(in_channels=1024, out_channels=512, kernel_size=(3, 3), padding=0),
            make_conv_layer(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=0),
            make_conv_layer(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=0),
            make_conv_layer(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0),
        ])

        self.conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))

        self.act = nn.ReLU()

        self.downscale = nn.MaxPool2d(kernel_size=2)

        self.upscale_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
        ])

        self.output_layer = make_conv_layer(in_channels=64, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        h = [] # store residual and skip connection

        for idx, layer in enumerate(self.down_layers):
            x = layer(x)

            if idx <= 3:
                h.append(x)
                x = self.downscale(x)
            
        for idx, layer in enumerate(self.up_layers):
            upscale_layer = self.upscale_layers[idx]
            
            c = h.pop() # stack 🤯

            x = upscale_layer(x) # conv

            c = center_crop(c, x)

            x = torch.cat((x, c), dim=1)
            x = layer(x)

        x = self.conv(x)

        return x


model = UNet()
test = torch.randn((4, 1, 572, 572))
res = model(test)

print(res, res.shape)