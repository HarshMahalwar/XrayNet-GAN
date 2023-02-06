import torch
import torch.nn as nn
import torchvision
from django.utils.baseconv import base64
from rest_framework.response import Response
from rest_framework.views import APIView
import base64
from rest_framework.parsers import MultiPartParser, FormParser
#
# from core.models import File
# from core.serializers import FileSerializer


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 128
FEATURES_GEN = 64


GenTrainedPneumonia = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
GenTrainedPneumonia.load_state_dict(torch.load('/home/harsh/Desktop/my_proj/XrayNet-GAN-Deployed/XrayNetGAN/staticfiles/Gen-x1r.pth', map_location='cpu'))

GenTrainedNormal = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
GenTrainedNormal.load_state_dict(torch.load('/home/harsh/Desktop/my_proj/XrayNet-GAN-Deployed/XrayNetGAN/staticfiles/Gen-x1r.pth', map_location='cpu'))


image_path = "/home/harsh/Desktop/my_proj/XrayNet-GAN-Deployed/XrayNetGAN/staticfiles/Saved.jpg"


def getImagePneumonia():
    cur_batch_size = 64
    noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
    GenTrainedPneumonia.eval()
    fake = GenTrainedPneumonia(noise)
    img_grid_fake = torchvision.utils.make_grid(
        fake[:32], normalize=True
    )
    torchvision.utils.save_image(img_grid_fake, image_path)


def getImageNormal():
    cur_batch_size = 64
    noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
    GenTrainedNormal.eval()
    fake = GenTrainedNormal(noise)
    img_grid_fake = torchvision.utils.make_grid(
        fake[:32], normalize=True
    )
    torchvision.utils.save_image(img_grid_fake, image_path)


class FileViewPneumonia(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        getImagePneumonia()
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        return Response(image_data)


class FileViewNormal(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        getImageNormal()
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        return Response(image_data)


