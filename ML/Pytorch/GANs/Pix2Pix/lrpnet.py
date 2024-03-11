import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LRPNet(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(LRPNet, self).__init__()
        # 3 256 256 -> 64 128 128
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )

        # Adding Residual Blocks after each downsample block
        self.res1 = ResidualBlock(features * 2)
        self.res2 = ResidualBlock(features * 4)
        self.res3 = ResidualBlock(features * 8)
        self.res4 = ResidualBlock(features * 8)

        # Adding Channel Attention after each downsample block
        self.ca1 = ChannelAttention(features * 2)
        self.ca2 = ChannelAttention(features * 4)
        self.ca3 = ChannelAttention(features * 8)
        self.ca4 = ChannelAttention(features * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8, features * 2, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 4, features, down=False, act="relu", use_dropout=False
        )

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x == 3 256 256
        d1 = self.initial_down(x) # 64 128 128
        d2 = self.ca1(self.res1(self.down1(d1))) # 128 64 64
        d3 = self.ca2(self.res2(self.down2(d2))) # 256 32 32
        d4 = self.ca3(self.res3(self.down3(d3))) # 512 16 16
        bottleneck = self.bottleneck(d4) # 512 8 8
        up1 = self.up1(bottleneck) # 512 16 16
        up2 = self.up2(torch.cat([up1, d4], 1)) # 256 32 32
        up3 = self.up3(torch.cat([up2, d3], 1)) # 128 64 64
        up4 = self.up4(torch.cat([up3, d2], 1)) # 64 128 128
        return self.final_up(torch.cat([up4, d1], 1)) # 3 256 256

def test():
    x = torch.randn((1, 3, 256, 256))
    model = LRPNet(in_channels=3, features=64)
    # preds = model(x)
    # print(preds.shape)
    torch.save(model, "model.pth")

if __name__ == "__main__":
    test()
