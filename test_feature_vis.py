import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 6, 2)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.decoder(x)
        return x


dataset = datasets.MNIST(
    root='PATH',
    transform=transforms.ToTensor(),
    download=True
)
loader = DataLoader(
    dataset,
    num_workers=2,
    batch_size=8,
    shuffle=True
)


model = Extractor()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 1
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)  # anet output인 x
        loss = criterion(output, data)  # vanishingNet에서 loss(x,y)
        loss.backward()
        optimizer.step()

        #print('Epoch {}, Batch idx {}, loss {}'.format(
        #    epoch, batch_idx, loss.item()))


def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img


# Plot some images
idx = torch.randint(0, output.size(0), ())
pred = normalize_output(output[idx, 0])
img = data[idx, 0]

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(img.detach().numpy())
axarr[1].imshow(pred.detach().numpy())

###############################################################
# Visualize conv filter
kernels = model.conv1.weight.detach()
fig, axarr = plt.subplots(kernels.size(0))
for idx in range(kernels.size(0)):
    axarr[idx].imshow(kernels[idx].squeeze())

from torchvision.utils import make_grid

kernels = model.extractor[0].weight.detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels)
plt.imshow(img.permute(1, 2, 0))

###############################################################


# Visualize feature maps
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

'''
model.conv1.register_forward_hook(get_activation('conv1'))
data, _ = dataset[0]
data.unsqueeze_(0)
output = model(data)

act = activation['conv1'].squeeze()
fig, axarr = plt.subplots(act.size(0))
for idx in range(act.size(0)):
    axarr[idx].imshow(act[idx])
    plt.imshow(act[idx])
'''

model.extractor[0].register_forward_hook(get_activation('ext_conv1'))
x = torch.randn(1, 3, 96, 96)
output = model(x)
print(output.shape)
# > torch.Size([1, 3, 96, 96])

act = activation['ext_conv1'].squeeze()
num_plot = 4
fig, axarr = plt.subplots(min(act.size(0), num_plot))
for idx in range(min(act.size(0), num_plot)):
    axarr[idx].imshow(act[idx])