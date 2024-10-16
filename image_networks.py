import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pil_video import make_video

import numpy as np


def Simple_NN_Model(input_size, hidden_layer_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, output_size),
        nn.Sigmoid(),
    )


def fourier_feature_mapping(coords, mapping, num_features=128, scale=10.0):
    # coords: Tensor of shape (batch_size, d), e.g., (batch_size, 2) for (x, y) coordinates
    # num_features: The dimensionality of the Fourier feature space
    # scale: The scale factor to control the range of frequencies in the Gaussian matrix B

    B = mapping
    # Step 1: Create a random frequency matrix B with shape (d, num_features)
    # Scale by 2π for periodicity and to control frequency range

    # Step 2: Transform the coordinates to higher-dimensional space using matrix B
    # (batch_size, d) x (d, num_features) -> (batch_size, num_features)
    transformed_coords = coords @ B

    # Step 3: Apply sinusoidal functions to create Fourier features
    # Stack the sin and cos transformations: (batch_size, 2 * num_features)
    fourier_features = torch.cat(
        [
            torch.sin(2 * np.pi * transformed_coords),
            torch.cos(2 * np.pi * transformed_coords),
        ],
        dim=-1,
    )

    return fourier_features


class SimpleTrainableImage:
    def __init__(self, dims, hidden_layer_size, num_features, scale):
        self.w = dims[0]
        self.h = dims[1]
        self.model = Simple_NN_Model(128 * 2, hidden_layer_size, 3)
        self.loss_fn = nn.BCELoss()
        self.optim = optim.Adam(self.model.parameters(), lr=0.01)
        self.num_features = num_features
        self.scale = scale
        self.mapping = fourier_feature_map(self.get_coords(), num_features, scale)

    def get_image(self):
        coords = []
        for col in range(0, self.w):
            for row in range(0, self.h):
                coords.append([col / self.w, row / self.h])
        coords = torch.tensor(coords, dtype=torch.float32)
        coords = fourier_feature_mapping(coords, self.mapping)
        im = self.model(coords).tolist()
        image = Image.new("RGB", (self.w, self.h), "black")
        pixels = image.load()
        for x in range(0, self.w):
            for y in range(0, self.h):
                c = im[x * self.h + y]
                pixels[x, y] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

        return image

    def get_coords(self):

        coords = []
        for x in range(0, self.w):
            for y in range(0, self.h):
                coords.append([x / self.w, y / self.h])
        return torch.tensor(coords, dtype=torch.float32)

    def train(self, target_func):
        coords = []
        target = []
        for x in range(0, self.w):
            for y in range(0, self.h):
                target.append(target_func(x, y))
                coords.append([x / self.w, y / self.h])
        coords = torch.tensor(coords, dtype=torch.float32)
        coords = fourier_feature_mapping(coords, self.mapping)
        pred = self.model(coords)
        target = torch.tensor(target, dtype=torch.float32)
        loss = self.loss_fn(pred, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        print("finished training step")

    def training_video(self, target_func, epochs):
        frames = []
        for _ in range(0, epochs):
            self.train(target_func)
            frames.append(self.get_image())
        make_video(frames, fps=5, delete_folder=False, play_video=False)


def fourier_feature_map(coords, num_features, scale):
    return torch.randn(coords.shape[-1], num_features) * scale


if __name__ == "__main__":

    # tests
    im = SimpleTrainableImage((200, 200), 100, 128, 10.0)
    target_im = Image.open("dog.jpg").resize((200, 200)).load()

    def color(x, y):
        p = target_im[x, y]
        return (p[0] / 255, p[1] / 255, p[2] / 255)

    # im.training_video(color, 100)
    for _ in range(0, 10):
        im.train(color)

    im.get_image().show("test")
