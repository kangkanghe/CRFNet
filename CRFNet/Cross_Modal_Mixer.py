import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
import math

#######################################ResNeXt##############################################

__all__ = ['ResNeXt']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(x.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print(residual.shape)
        out += residual
        out = self.relu(out)
        # print(out.shape)
        return out

class ResNeXt(nn.Module):

    def __init__(self, in_C, out_C, block, layers, num_classes=3, num_group=32):
        self.inplanes = out_C
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(in_C, out_C, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_C)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, out_C, layers[0], num_group)
        # self.avgpool = nn.AvgPool2d(8, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x_layer1 = self.layer1(x)
        return x_layer1
def resnext50_3(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(7, 64, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
def resnext50_10(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(10, 64, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
def resnext50_4(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(4, 64, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

#######################################MMF_MLPMixer2#############################################
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim=256, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.dim = dim
        self.reduction = 1
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2

class PatchEmbeddings(nn.Module):

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        channels: int
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GlobalAveragePooling(nn.Module):

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class Classifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.model.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MLPBlockFuse(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MixerBlock(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )
        self.token_mixing_sv = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_sv = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )

    def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        # print(img.shape)
        # print(self.token_mixing_img(img).shape)
        x_token_img = img + self.token_mixing_img(img)
        # print(x_token_img.shape, self.token_mixing_img(img).shape)
        x_img = x_token_img + self.channel_mixing_img(x_token_img)
        # print(x_img.shape)

        x_token_sv = sv + self.token_mixing_sv(sv)
        # print(x_token_sv.shape)
        x_sv = x_token_sv + self.channel_mixing_sv(x_token_sv)
        # print(x_sv.shape)
        return x_img, x_sv

class MixerBlock_fuse(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*2),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*2),
            MLPBlock(num_channels*2, channels_hidden_dim)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # print(img.shape)
        # print(self.token_mixing_img(img).shape)
        x_token_img = img + self.token_mixing_img(img)
        x_img = x_token_img + self.channel_mixing_img(x_token_img)
        # print(x_img.shape)
        return x_img

class MMF_MLPMixer2(nn.Module):     # 两模态

    def __init__(
            self,
            num_classes: int,
            image_size: int = 11,
            channels: int = 256,
            patch_size: int = 3,
            num_layers: int = 4,
            hidden_dim: int = 256,
            tokens_hidden_dim: int = 128,
            channels_hidden_dim: int = 1024
    ):
        super().__init__()
        # num_patches = (image_size // patch_size) ** 2
        num_patches = (((image_size - 2 - patch_size) // patch_size) + 1) ** 2
        # self.embed_landsat = PatchEmbeddings(patch_size, hidden_dim, channels)  # 三模态
        self.embed_geography = PatchEmbeddings(patch_size, hidden_dim, channels)
        self.embed_geology = PatchEmbeddings(patch_size, hidden_dim, channels)
        # self.embed_fuse = PatchEmbeddings(patch_size, hidden_dim, channels*2)
        self.Mixerlayer1 = MixerBlock(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
        )

        self.Mixerlayer_fuse_early = MixerBlock_fuse(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
        )

        # self.norm = nn.LayerNorm(hidden_dim * 6)  # 三模态
        self.norm = nn.LayerNorm(hidden_dim * 2)    # 两模态

        self.pool = GlobalAveragePooling(dim=1)

        # self.classifier = Classifier(hidden_dim * 6, num_classes)  # 三模态
        self.classifier = Classifier(hidden_dim * 2, num_classes)     # 两模态



    # def forward(self, landsat: torch.Tensor, geography: torch.Tensor, geology: torch.Tensor) -> torch.Tensor:  # 三模态
    def forward(self, landsat: torch.Tensor, all_factor: torch.Tensor) -> torch.Tensor:    # 两模态
        feature_rectify_module = FeatureRectifyModule(dim=256, reduction=1, lambda_c=.5, lambda_s=.5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_rectify_module = feature_rectify_module.to(device)
        landsat_rectify, all_factor_rectify = feature_rectify_module(landsat, all_factor)  #（256,256,17,17）
        # x_landsat = self.embed_landsat(landsat)  # 三模态  # [b, p, c]
        landsat_embed = self.embed_geography(landsat_rectify)  # [256,25,256]
        all_factor_embed = self.embed_geology(all_factor_rectify)
        # x_landsat, x_all_factor = FeatureRectifyModule(x_landsat, x_all_factor)

        landsat_Mixer, geology_Mixer = self.Mixerlayer1(landsat_embed, all_factor_embed)  # 两模态 [b, p, c]
        x_fuse_late = torch.cat([landsat_Mixer, geology_Mixer], 2)
        x_fuse_late = self.Mixerlayer_fuse_early(x_fuse_late)

        # x = torch.cat([x_fuse_early, x_fuse_ontime, x_fuse_late], 2)  # 三模态
        x = x_fuse_late            # 两模态[256,25,512]
        x = self.norm(x)
        x = self.pool(x)  # [b, c]
        # print(x.shape)
        x = self.classifier(x)  # [b, num_classes]
        return x

class MMFMixer(nn.Module):
    def __init__(self, n_class, image_size, hidden_dim=256):
        super(MMFMixer, self).__init__()
        self.n_class = n_class
        self.resnext50_3 = resnext50_3()  # landsat
        self.resnext50_10 = resnext50_10()  # geography
        self.mixer2 = MMF_MLPMixer2(num_classes=3, image_size=image_size, channels=256)

        # 假设每个通道的height和width都是5
        prompt_shape = (1, 1, image_size, image_size)  # 适用于单个特征通道的提示向量形状
        # 初始化提示向量
        self.prompt_vector_missing_construction = nn.Parameter(torch.randn(*prompt_shape))
        self.prompt_vector_missing_geology = nn.Parameter(torch.randn(*prompt_shape))
        self.prompt_vector_missing_RCI = nn.Parameter(torch.randn(*prompt_shape))

        # 将提示向量初始化为正态分布
        trunc_normal_(self.prompt_vector_missing_construction, std=.02)
        trunc_normal_(self.prompt_vector_missing_geology, std=.02)
        trunc_normal_(self.prompt_vector_missing_RCI, std=.02)

    def forward(self, variable1=None, variable2=None, variable3=None):      # 三模态
        # missing_modalities 是一个字典，包含了缺失模态的信息
        # 例如：{"construction": True, "geology": False, "RCI": True}

        if variable3 == 'train':
            landsat = self.resnext50_3(variable1)
            all_factor = self.resnext50_10(variable2)
            patch_mixer = self.mixer2(landsat, all_factor)

            # 将提示向量扩展到批量大小
            prompt_missing = self.prompt_vector_missing_construction.expand_as(variable2[:, 7:10, :, :])
            # 计算提示向量
            # 如果模态缺失，则使用对应的提示向量
            variable2_modified = variable2.clone()  # 克隆variable2以避免修改原始数据
            variable2_modified[:, 7:, :, :] = prompt_missing
            all_factor_mission = self.resnext50_10(variable2_modified)
            patch_mixer_mission = self.mixer2(landsat, all_factor_mission)

            return patch_mixer, patch_mixer_mission

        elif variable3 == 'val':
            landsat = self.resnext50_3(variable1)
            # 将提示向量扩展到批量大小
            prompt_missing = self.prompt_vector_missing_construction.expand_as(variable2[:, 7:10, :, :])
            # 计算提示向量
            # 如果模态缺失，则使用对应的提示向量
            variable2_modified = variable2.clone()  # 克隆variable2以避免修改原始数据
            variable2_modified[:, 7:, :, :] = prompt_missing
            all_factor_mission = self.resnext50_10(variable2)
            patch_mixer_mission = self.mixer2(landsat, all_factor_mission)

            return patch_mixer_mission