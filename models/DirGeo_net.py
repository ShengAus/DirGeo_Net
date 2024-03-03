import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(MLP, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = torch.mean(x, dim=(2,3,4)) # (N,C)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))

        y = y.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # print(x.shape, y.shape)
        out = x * y
        return out

class MLPV2(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(MLPV2, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x, grad):

        y = torch.mean(x, dim=(2,3,4)) # (N,C)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))

        y = y + grad
        y = y.unsqueeze(2).unsqueeze(3).unsqueeze(4) # (N, C, 1, 1, 1)
        # print(x.shape, y.shape)
        out = x * y # (N, C, W, H, D)

        return out, y.squeeze(4).squeeze(3).squeeze(2)
    
class GradEncoding(nn.Module):

    def __init__(self, n_features, out_features, hidden_features = None):
        super(GradEncoding, self).__init__()
        hidden_features = hidden_features or out_features // 16
        self.linear1 = nn.Linear(n_features, hidden_features)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, grad):
        # grad (N,C)
        out = self.nonlin1(self.linear1(grad)) # (N, C//16)
        out = self.nonlin2(self.linear2(out))  # (N, C)

        return out

class Conv(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Conv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        x = self.conv3d(input)

        # gated features
        if self.batch_norm:
            x = self.batch_norm3d(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

class DeConv(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size = 3
            , stride = 2, padding=1, output_padding = 1, dilation=1
            , groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(DeConv, self).__init__()

        self.conv3d2 = nn.Sequential(
                        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride= stride
                            , padding= padding, output_padding = output_padding, dilation =dilation),  
                        nn.BatchNorm3d(out_channels),
                        activation)

    def forward(self, input):
        x = self.conv3d2(input)
        return x

class DirGeonet(nn.Module):
    def __init__(self, in_channels=8, out_channels=6, batch_norm=True, cnum=32):
        super(DirGeonet, self).__init__()
        print("SeUnet is created")
        self.in_channels = in_channels
        self.out_channels = out_channels
        activation = nn.ReLU(inplace=True)
        # activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling In_dim
        self.enc1_1 = Conv(in_channels, cnum, 3, 1, padding= 1, batch_norm=batch_norm, activation=activation)
        self.enc1_2 = Conv(cnum, cnum, 3, 2, padding= 1, batch_norm=batch_norm, activation=activation)
        self.se1 = MLPV2(cnum)
        self.grad_embed0 = nn.Linear(3, out_features=1)
        self.activation = activation
        self.grad_embed1 = GradEncoding(n_features=7, out_features=cnum)
        # In_dim/2
        self.enc2_1 = Conv(cnum, 2 * cnum, 3, 1, padding= 1, batch_norm=batch_norm, activation=activation)
        self.enc2_2 = Conv(2 * cnum, 2 * cnum, 3, 2, padding=1, batch_norm=batch_norm, activation=activation)
        self.se2 = MLPV2(2 * cnum)
        self.grad_embed2 = GradEncoding(n_features=cnum, out_features=2 * cnum)
        # In_dim/4
        self.enc3_1 = Conv(2 * cnum, 4 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.enc3_2 = Conv(4 * cnum, 4 * cnum, 3, 2, padding=1, batch_norm=batch_norm, activation=activation)
        self.se3 = MLPV2(4 * cnum)
        self.grad_embed3 = GradEncoding(n_features=2 * cnum, out_features=4 * cnum)
        # In_dim/8
        self.enc4_1 = Conv(4 * cnum, 8 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.enc4_2 = Conv(8 * cnum, 8 * cnum, 3, 2, padding=1, batch_norm=batch_norm, activation=activation)
        self.se4 = MLPV2(8 * cnum)
        self.grad_embed4 = GradEncoding(n_features=4 * cnum, out_features=8 * cnum)
    
        # Bridge In_dim/16
        self.bridge = Conv(8 * cnum, 16 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)

        # Up sampling In_dim/16
        self.dec1_1 = DeConv(2, 16 * cnum, 8 * cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec1_2 = Conv(16 * cnum, 8 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.se_de1 = MLP(8 * cnum)

        # Up Sampling In_dim/8
        self.dec2_1 = DeConv(2, 8 * cnum, 4 * cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec2_2 = Conv(8 * cnum, 4 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.se_de2 = MLP(4 * cnum)

        # Up Sampling In_dim/4
        self.dec3_1 = DeConv(2, 4 * cnum, 2 * cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec3_2 = Conv(4 * cnum, 2 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.se_de3 = MLP(2 * cnum)

        # Up Sampling In_dim/2
        self.dec4_1 = DeConv(2, 2 * cnum, cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec4_2 = Conv(2 * cnum, cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.se_de4 = MLP(cnum)

        # Output In_dim
        self.out = Conv(cnum, out_channels, 3, 1, padding=1, batch_norm=False, activation=None)
        self.out2 = nn.Linear(out_channels, out_channels)
        self.se_out = MLP(out_channels, reduction=1)

    def forward(self, x, grad, encoder_only=False):

        feat = []

        # x: b c w h d
        # Down sampling
        down_1 = self.enc1_1(x)
        pool_1 = self.enc1_2(down_1)
        grad_0 = self.activation(torch.flatten(self.grad_embed0(grad),start_dim=1))
        # print(grad_0.shape)
        grad_1 = self.grad_embed1(grad_0) # (N, C)
        pool_1_se, pool_1_grad = self.se1(pool_1, grad_1)

        down_2 = self.enc2_1(pool_1_se)
        pool_2 = self.enc2_2(down_2)
        grad_2 = self.grad_embed2(pool_1_grad) # (N, 2C)
        pool_2_se, pool_2_grad = self.se2(pool_2, grad_2)
            
        down_3 = self.enc3_1(pool_2_se)
        pool_3 = self.enc3_2(down_3)
        grad_3 = self.grad_embed3(pool_2_grad) # (N, 4C)
        pool_3_se, pool_3_grad = self.se3(pool_3, grad_3)
            
        down_4 = self.enc4_1(pool_3_se)
        pool_4 = self.enc4_2(down_4)
        grad_4 = self.grad_embed4(pool_3_grad) # (N, 4C)
        pool_4_se, pool_4_grad  = self.se4(pool_4, grad_4)
            
        # print('pool shape', pool_1.shape, pool_2.shape, pool_3.shape, pool_4.shape)
        if encoder_only:
            return feat

        # Bridge
        bridge = self.bridge(pool_4_se)

        # Up sampling
        trans_1 = self.dec1_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.dec1_2(concat_1)
        up_1_se = self.se_de1(up_1)

        trans_2 = self.dec2_1(up_1_se)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.dec2_2(concat_2)
        up_2_se = self.se_de2(up_2)

        trans_3 = self.dec3_1(up_2_se)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.dec3_2(concat_3)
        up_3_se = self.se_de3(up_3)

        trans_4 = self.dec4_1(up_3_se)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.dec4_2(concat_4)
        up_4_se = self.se_de4(up_4)
 
        out = self.out(up_4_se)
        out2 = self.out2(out.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        out_se = self.se_out(out2)

        return out_se

