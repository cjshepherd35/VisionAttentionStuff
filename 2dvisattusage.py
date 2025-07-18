import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visionatt import Transform
num_epochs = 2
num_inputs = 256
num_hidden = 512
block_size = 16
batch_size = 64
num_outputs = 200
num_layers = 6
numfeats = 768

data_dir = "C:/Users/cshep/virtenv/threeDAttention/tiny-imagenet-200"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282])
])

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def patchify(imgs, patch_size):
    """
    imgs: Tensor of shape (B, C, H, W)
    Returns: (B, N, patch_dim) where N = num patches
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image must be divisible by patch size"
    
    h_patches = H // patch_size
    w_patches = W // patch_size
    N = h_patches * w_patches

    # Split into patches
    patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Shape: (B, C, h_patches, w_patches, P, P)
    
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, h_p, w_p, C, P, P)
    patches = patches.flatten(1, 2)              # (B, N, C, P, P)
    patch_dim = C * patch_size * patch_size
    patches = patches.reshape(B, N, patch_dim)   # (B, N, patch_dim)
    
    return patches

tr = Transform(num_inputs, num_hidden, num_outputs, block_size,num_layers)
lin = nn.Linear(numfeats, num_inputs)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(tr.parameters(), lr=0.01)
i = 0
for j in range(num_epochs):
    for batch_idx, (features, labels) in enumerate(train_loader):
        patches = patchify(features, patch_size=block_size)
        xbatch = lin(patches)
        # print(patches.shape)
        # print("x ", xbatch.size())
        output = tr(xbatch)
        loss = loss_fn(output, labels)
        
        if (i%20 == 0):
            print('i ' , i, 'loss ', loss)

        i += 1

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
        opt.step()
