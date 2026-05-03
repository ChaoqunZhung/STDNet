import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.cfg.VideoSwinTransformer import SwinTransformer3D
from lib.models.cfg.neck import AddFusion3D

class TimeSpaceExchange(nn.Module):
    def __init__(
        self,
        in_channels: int,
        t_depth: int,
        upscale_factor: int = 4,
        out_channels: int = 48,
    ):
        super().__init__()

        flat_channels = in_channels * t_depth
        hidden_channels = out_channels * (upscale_factor ** 2)

        self.project = nn.Sequential(
            nn.Conv2d(flat_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.final_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.view(b, c * t, h, w)
        x = self.project(x)
        x = self.shuffle(x)
        x = self.final_conv(x)
        return x


class Dynamic_Stream(nn.Module):
    def __init__(
        self,
        base_channels: int = 48,
        depths=(2, 2, 6, 2),
        fusion_t_size: int = 8,
        upscale_factor: int = 4,
        out_channels: int = 48,
    ):
        super().__init__()

        self.channels = [base_channels * 2**i for i in range(len(depths))]

        self.transformer = SwinTransformer3D(
            embed_dim=self.channels[0],
            window_size=(5, 7, 7),
            depths=depths,
        )

        self.fusion = AddFusion3D(
            self.channels[0],
            self.channels,
            up_f=2,
        )

        self.fusion_t_size = fusion_t_size

        self.ts_exchange = TimeSpaceExchange(
            in_channels=self.channels[0],
            t_depth=self.fusion_t_size,
            upscale_factor=upscale_factor,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.transformer(x)
        fusioned = self.fusion(features, 0, len(self.channels))

        if fusioned.shape[2] != self.fusion_t_size:
            fusioned = F.interpolate(
                fusioned,
                size=(
                    self.fusion_t_size,
                    fusioned.shape[3],
                    fusioned.shape[4],
                ),
                mode="trilinear",
                align_corners=False,
            )

        output = self.ts_exchange(fusioned)
        return output


if __name__ == "__main__":
    model = Dynamic_Stream().cuda()
    x = torch.randn(2, 3, 15, 512, 512).cuda()
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")