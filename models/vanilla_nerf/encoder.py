"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.append('/home/zubairirshad/nerf_pl')
from models.vanilla_nerf.util import get_norm_layer
# from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler
# from inplace_abn import InPlaceABN
import torch.nn as nn

# def spatial_encoder_index(uv, latent, mode = 'bilinear', padding='zeros'):
#     """
#     Get pixel-aligned image features at 2D image coordinates
#     :param uv (B, N, 2) image points (x,y)
#     :param cam_z ignored (for compatibility)
#     :param image_size image size, either (width, height) or single int.
#     if not specified, assumes coords are in [-1, 1]
#     :param z_bounds ignored (for compatibility)
#     :return (B, L, N) L is latent size
#     """
#     # with profiler.record_function("encoder_index"):
#     #     if uv.shape[0] == 1 and self.latent.shape[0] > 1:
#     #         uv = uv.expand(self.latent.shape[0], -1, -1)

#         # with profiler.record_function("encoder_index_pre"):
#         #     if len(image_size) > 0:
#         #         if len(image_size) == 1:
#         #             image_size = (image_size, image_size)
#         #         scale = self.latent_scaling / image_size
#         #         uv = uv * scale - 1.0

#     uv = uv.unsqueeze(2)  # (B, N, 1, 2)
#     samples = F.grid_sample(
#         latent,
#         uv,
#         align_corners=True,
#         mode=mode,
#         padding_mode=padding
#     )
#     return samples[:, :, :, 0]  # (B, C, N)


norm_layer = get_norm_layer("batch")

pretrained_model = getattr(torchvision.models, "resnet34")(
    pretrained=True, norm_layer=norm_layer
)

class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)
    
class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)

class ResUNet(nn.Module):
    def __init__(self, out_ch = 64):
        super().__init__()
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        self.maxpool = pretrained_model.maxpool
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        filters = [64, 128, 256, 512]

        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)

        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)
        # self.layer4 = pretrained_model.layer4
        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)
        self.latent_size = 32
    
    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        print("x3", x3.shape)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)
        print("x", x.shape)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)
        print("x", x.shape)

        x_out = self.out_conv(x)
        return x_out
    
class CustomResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        self.maxpool = pretrained_model.maxpool
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        # self.layer4 = pretrained_model.layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x
    
class FeatureAggregator(nn.Module):
    def __init__(self, input_ch=512, output_ch = 128):
        super(FeatureAggregator, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch*2, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_ch*2, output_ch, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = CustomResNet34()
            self.feature_aggregator = FeatureAggregator()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        # x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            print("x", x.shape)
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            print("x", x.shape)
            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                print("x after first pool", x.shape)
                x = self.model.layer1(x)
                print("x after 1", x.shape)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                print("x after 2", x.shape)
                latents.append(x)
                
            if self.num_layers > 3:
                x = self.model.layer3(x)
                print("x after 3", x.shape)
                latents.append(x)
            # if self.num_layers > 4:
            #     x = self.model.layer4(x)
            #     latents.append(x)
            print("==========================\n\n\n")
            # self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            print("latents", latent_sz)
            for i in range(len(latents)):
                print("latents[i] before", latents[i].shape)
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
                print("latents[i] after", latents[i].shape)
            # self.latent = torch.cat(latents, dim=1)
            latent = torch.cat(latents, dim=1)

        # latent = self.feature_aggregator(latent)

        # self.latent_scaling[0] = self.latent.shape[-1]
        # self.latent_scaling[1] = self.latent.shape[-2]
        # print("self.latent_scaling", self.latent_scaling)
        # self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


# class SpatialEncoder(nn.Module):
#     """
#     2D (Spatial/Pixel-aligned/local) image encoder
#     """

#     def __init__(
#         self,
#         backbone="resnet34",
#         pretrained=True,
#         num_layers=4,
#         index_interp="bilinear",
#         index_padding="border",
#         upsample_interp="bilinear",
#         feature_scale=1.0,
#         use_first_pool=True,
#         norm_type="batch",
#     ):
#         """
#         :param backbone Backbone network. Either custom, in which case
#         model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
#         model from torchvision is used
#         :param num_layers number of resnet layers to use, 1-5
#         :param pretrained Whether to use model weights pretrained on ImageNet
#         :param index_interp Interpolation to use for indexing
#         :param index_padding Padding mode to use for indexing, border | zeros | reflection
#         :param upsample_interp Interpolation to use for upscaling latent code
#         :param feature_scale factor to scale all latent by. Useful (<1) if image
#         is extremely large, to fit in memory.
#         :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
#         features too much (ResNet only)
#         :param norm_type norm type to applied; pretrained model must use batch
#         """
#         super().__init__()

#         if norm_type != "batch":
#             assert not pretrained

#         self.use_custom_resnet = backbone == "custom"
#         self.feature_scale = feature_scale
#         self.use_first_pool = use_first_pool
#         norm_layer = get_norm_layer(norm_type)

#         if self.use_custom_resnet:
#             print("WARNING: Custom encoder is experimental only")
#             print("Using simple convolutional encoder")
#             self.model = ConvEncoder(3, norm_layer=norm_layer)
#             self.latent_size = self.model.dims[-1]
#         else:
#             print("Using torchvision", backbone, "encoder")
#             self.model = getattr(torchvision.models, backbone)(
#                 pretrained=pretrained, norm_layer=norm_layer
#             )
#             print("=========================\n\n\n")
#             print("actual model", self.model)
#             print("==================\n\n\n")
#             # Following 2 lines need to be uncommented for older configs
#             self.model.fc = nn.Sequential()
#             self.model.avgpool = nn.Sequential()
#             self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

#         self.num_layers = num_layers
#         self.index_interp = index_interp
#         self.index_padding = index_padding
#         self.upsample_interp = upsample_interp
#         # self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
#         # self.register_buffer(
#         #     "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
#         # )
#         # self.latent (B, L, H, W)

#     # def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
#     #     """
#     #     Get pixel-aligned image features at 2D image coordinates
#     #     :param uv (B, N, 2) image points (x,y)
#     #     :param cam_z ignored (for compatibility)
#     #     :param image_size image size, either (width, height) or single int.
#     #     if not specified, assumes coords are in [-1, 1]
#     #     :param z_bounds ignored (for compatibility)
#     #     :return (B, L, N) L is latent size
#     #     """
#     #     with profiler.record_function("encoder_index"):
#     #         if uv.shape[0] == 1 and self.latent.shape[0] > 1:
#     #             uv = uv.expand(self.latent.shape[0], -1, -1)

#     #         # with profiler.record_function("encoder_index_pre"):
#     #         #     if len(image_size) > 0:
#     #         #         if len(image_size) == 1:
#     #         #             image_size = (image_size, image_size)
#     #         #         scale = self.latent_scaling / image_size
#     #         #         uv = uv * scale - 1.0

#     #         uv = uv.unsqueeze(2)  # (B, N, 1, 2)
#     #         samples = F.grid_sample(
#     #             self.latent,
#     #             uv,
#     #             align_corners=True,
#     #             mode=self.index_interp,
#     #             padding_mode=self.index_padding,
#     #         )
#     #         return samples[:, :, :, 0]  # (B, C, N)

#     def forward(self, x):
#         """
#         For extracting ResNet's features.
#         :param x image (B, C, H, W)
#         :return latent (B, latent_size, H, W)
#         """
#         if self.feature_scale != 1.0:
#             x = F.interpolate(
#                 x,
#                 scale_factor=self.feature_scale,
#                 mode="bilinear" if self.feature_scale > 1.0 else "area",
#                 align_corners=True if self.feature_scale > 1.0 else None,
#                 recompute_scale_factor=True,
#             )
#         # x = x.to(device=self.latent.device)

#         if self.use_custom_resnet:
#             self.latent = self.model(x)
#         else:
#             x = self.model.conv1(x)
#             x = self.model.bn1(x)
#             x = self.model.relu(x)

#             latents = [x]
#             if self.num_layers > 1:
#                 if self.use_first_pool:
#                     x = self.model.maxpool(x)
#                 x = self.model.layer1(x)
#                 latents.append(x)
#             if self.num_layers > 2:
#                 x = self.model.layer2(x)
#                 latents.append(x)
#             if self.num_layers > 3:
#                 x = self.model.layer3(x)
#                 latents.append(x)
#             if self.num_layers > 4:
#                 x = self.model.layer4(x)
#                 latents.append(x)

#             # self.latents = latents
#             align_corners = None if self.index_interp == "nearest " else True
#             latent_sz = latents[0].shape[-2:]
#             for i in range(len(latents)):
#                 latents[i] = F.interpolate(
#                     latents[i],
#                     latent_sz,
#                     mode=self.upsample_interp,
#                     align_corners=align_corners,
#                 )
#             # self.latent = torch.cat(latents, dim=1)
#             latent = torch.cat(latents, dim=1)
#         # self.latent_scaling[0] = self.latent.shape[-1]
#         # self.latent_scaling[1] = self.latent.shape[-2]
#         # print("self.latent_scaling", self.latent_scaling)
#         # self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
#         return latent

#     @classmethod
#     def from_conf(cls, conf):
#         return cls(
#             conf.get_string("backbone"),
#             pretrained=conf.get_bool("pretrained", True),
#             num_layers=conf.get_int("num_layers", 4),
#             index_interp=conf.get_string("index_interp", "bilinear"),
#             index_padding=conf.get_string("index_padding", "border"),
#             upsample_interp=conf.get_string("upsample_interp", "bilinear"),
#             feature_scale=conf.get_float("feature_scale", 1.0),
#             use_first_pool=conf.get_bool("use_first_pool", True),
#         )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        
        self.model.fc = nn.Sequential()
        # self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
# class ConvBnReLU(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=3, stride=1, pad=1,
#                  norm_act=InPlaceABN):
#         super(ConvBnReLU, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size, stride=stride, padding=pad, bias=False)
#         self.bn = norm_act(out_channels)


#     def forward(self, x):
#         return self.bn(self.conv(x))

# from inplace_abn import InPlaceABN
import torch.nn as nn

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm3d):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))
    

class Volume3DCNN(nn.Module):
    def __init__(self, in_channels, norm_act=nn.BatchNorm3d):
        super(Volume3DCNN, self).__init__()
        self.convdown0 = ConvBnReLU3D(in_channels, 64, norm_act=norm_act)
        self.convdown1 = ConvBnReLU3D(64, 32, norm_act=norm_act)
        self.convdown2 = ConvBnReLU3D(32, 16, norm_act=norm_act)
        self.convdown3 = ConvBnReLU3D(16, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.convdown3(self.convdown2(self.convdown1(self.convdown0(x))))
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x
    
# class FeatureNet(nn.Module):
#     """
#     output 3 levels of features using a FPN structure
#     """
#     def __init__(self, norm_act=nn.BatchNorm2d):
#         super(FeatureNet, self).__init__()

#         self.conv0 = nn.Sequential(
#                         ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
#                         ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

#         self.conv1 = nn.Sequential(
#                         ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
#                         ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
#                         ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

#         self.conv2 = nn.Sequential(
#                         ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
#                         ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
#                         ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

#         self.toplayer = nn.Conv2d(32, 32, 1)
#         self.latent_size = 32

#     def _upsample_add(self, x, y):
#         return F.interpolate(x, scale_factor=2,
#                              mode="bilinear", align_corners=True) + y

#     def forward(self, x):
#         # x: (B, 3, H, W)
#         x = self.conv0(x) # (B, 8, H, W)
#         x = self.conv1(x) # (B, 16, H//2, W//2)
#         x = self.conv2(x) # (B, 32, H//4, W//4)
#         x = self.toplayer(x) # (B, 32, H//4, W//4)

#         return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class MultiViewTransformer(nn.Module):
    def __init__(self,feature_dim=128, num_heads=2, dropout=0.2):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = feature_dim
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=num_heads,
                dim_feedforward=2*self.feature_dim,
                dropout=dropout
            ),
            num_layers=2
        )
        
        
    def forward(self, features, masked_features):
        # Create attention mask
        attention_mask = ~torch.all(masked_features == 0, dim=1)
        attention_mask = attention_mask.bool()
        
        # Apply transformer
        features = self.transformer(features, attention_mask)
        
        # Average over views
        features = features.mean(dim=1)

        return features



if __name__ == "__main__":

    # encoder = ResUNet()

    # total_params = sum(p.numel() for p in encoder.parameters())
    # print("total custom  pretrained params", total_params)

    # img = torch.randn((3,3,224,224))
    # latent = encoder(img)
    # print("custom resent 34 ibrnet unet",latent.shape)

    encoder = SpatialEncoder()

    total_params = sum(p.numel() for p in encoder.parameters())
    print("total spatial encoder pretrained params", total_params)

    img = torch.randn((3,3,240,320))
    latent = encoder(img)
    print(latent.shape)

    # attention_layer = nn.MultiheadAttention(128, 4, dropout=0.2)
    # # mha = MultiHeadAttention(2, 16, 2, 2, 0.1)

    # a = torch.randn((3, 385000, 128))
    # b = attention_layer(a, a, a)[0]
    # print(b.shape)
    # # print("==============================\n\n\n")
    # # print("Conv3D\n\n\n")
    # # print("==============================\n\n\n")

    # conv3d = Volume3DCNN(in_channels=128, norm_act=nn.BatchNorm3d)

    # total_params = sum(p.numel() for p in conv3d.parameters())
    # print("total conv3d params", total_params)


    # feature_volume = torch.randn((3,128,96,96,96))
    # print("feature_volume", feature_volume.shape)

    # out_conv3d = conv3d(feature_volume)

    # print("out conv3d",out_conv3d.shape)

    # encoder_layer = nn.TransformerEncoderLayer(64, 2, 256)
    # transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)

    # # transformer_encoder = MultiViewTransformer()

    # # mask = torch.randn((1,3,64,64,64,128)) > 0.5
    # # mask = mask.view(3,-1,128)

    # # total_params = sum(p.numel() for p in transformer_encoder.parameters())
    # # print("total spatial encoder pretrained params", total_params)

    # features = torch.randn((1,64,64,64,3,64))

    # features = features.view(-1,3,64)

    # encoded = transformer_encoder(features)
    # print(encoded.shape)
    # encoded = encoded.view(1,3,64,64,64,64)
    # print(encoded.shape)

    # for name, param in encoder.named_parameters():
    #     # if "layer4" in name:
    #     print(f'{name} requires_grad: {param.requires_grad}')
    # print("===============\n\n\n")
    # print(encoder)
    # print("=========================\n\n\n")
    # print("self.latent_scaling", self.latent_scaling)