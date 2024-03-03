import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from collections.abc import Sequence
from torch.nn import LayerNorm
from models.SWIN_trans import SwinTransformer, PatchMergingV2

class UnetrBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1):

      super().__init__()

      self.layer1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding = 1)
      self.norm1 = nn.BatchNorm3d(out_channels)
      self.layer2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding = 1)
      self.norm2 = nn.BatchNorm3d(out_channels)
      self.act = nn.ReLU()

    def forward(self,x):
      shortcut = x
      x1 = self.act(self.norm1(self.layer1(x)))
      x2 = self.act(self.norm2(self.layer2(x1)))

      return x2

class UnetrUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsample_stride = 2, stride = 1):

        super().__init__()
        self.convTran = nn.Sequential(
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size, upsample_stride
                        , padding= 1, output_padding = 1, dilation = 1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU())

        self.layer2 = UnetrBasicBlock(out_channels*2, out_channels, kernel_size, stride)


    def forward(self, x , hidden_state):
        x = self.convTran(x)
        x = self.layer2(torch.cat([x, hidden_state], dim = 1))

        return x

class SWINUNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        window_size: int = 3,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (2, 4, 8, 12),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:

        super().__init__()

        patch_sizes = (patch_size,)*spatial_dims
        window_size = (window_size, )*spatial_dims

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=PatchMergingV2,
            use_v2=use_v2
        )

        self.encoder1 = UnetrBasicBlock(
            in_channels = in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1
        )

        self.encoder2 = UnetrBasicBlock(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1
        )

        self.encoder3 = UnetrBasicBlock(
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1
        )

        self.encoder4 = UnetrBasicBlock(
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1
        )

        self.encoder10 = UnetrBasicBlock(
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1
        )

        self.decoder5 = UnetrUpBlock(
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_stride=2,
            stride = 1
        )

        self.decoder4 = UnetrUpBlock(
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_stride=2,
            stride = 1
        )

        self.decoder3 = UnetrUpBlock(
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_stride=2,
            stride = 1
        )
        self.decoder2 = UnetrUpBlock(
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_stride=2,
            stride = 1

        )

        self.decoder1 = UnetrUpBlock(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_stride=2,
            stride = 1
        )
        self.out = nn.Conv3d(feature_size, out_channels, 3, 1, padding = 1)                   
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        # for i in hidden_states_out:
        #   print(i.shape)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        logits = self.linear(logits.permute(0,2,3,4,1)).permute(0,4,1,2,3)

        return logits