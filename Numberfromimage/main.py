import torch
from torch import nn, save, load
from torchvision.transforms import ToTensor
import gradio as gr
import numpy as np
from PIL import Image


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)


clf = ImageClassifier().to('cpu')


with open('model_state.pt', 'rb') as f:
    clf.load_state_dict(torch.load(f, map_location=torch.device('cpu')))


def recognize_digit(canv):
    if canv is not None:

        img = canv
        img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
        prediction = clf(img_tensor)
        return (prediction.argmax().item())

    else:
        return "sorry"


def hi(canv):
    img = Image.fromarray(canv)
    size = img.size
    return f"Image size: {size}"


iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(image_mode='L', sources='upload'),

    outputs=gr.Text(),
    live=True
)
iface.launch()
