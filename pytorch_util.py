import io
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models
import dowload_model

NB_CLASS = 10

# load model
alexnet = models.alexnet(pretrained=True)
device = torch.device('cpu')
alexnet.classifier[6] = nn.Linear(4096, NB_CLASS)
alexnet.load_state_dict(torch.load("what_is_that_cloud_alexnet.pt", map_location=device))
alexnet.eval()

mobilenet = models.mobilenet_v2(pretrained=True)
device = torch.device('cpu')
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, NB_CLASS)
mobilenet.load_state_dict(torch.load("what_is_that_cloud_mobilenet_v2.pt", map_location=device))
mobilenet.eval()

squeezenet = models.squeezenet1_0(pretrained=True)
device = torch.device('cpu')
squeezenet.classifier[1] = nn.Conv2d(512, NB_CLASS, kernel_size=(1, 1), stride=(1, 1))
squeezenet.load_state_dict(torch.load("what_is_that_cloud_squeezenet.pt", map_location=device))
squeezenet.eval()


# image -> tensor
def transform_image(image_bytes):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([  # T.Grayscale(num_output_channels=1),
        T.Resize((512, 512)),
        T.ToTensor(),
        normalize,
    ])

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


# predict
def get_alexnet_prediction(image_tensor):
    outputs = alexnet(image_tensor)
    return outputs


def get_mobilenet_prediction(image_tensor):
    outputs = mobilenet(image_tensor)
    return outputs


def get_squeezenet_prediction(image_tensor):
    outputs = squeezenet(image_tensor)
    return outputs
