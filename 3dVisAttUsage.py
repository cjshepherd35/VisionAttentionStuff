import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visionatt import Transform
import cv2 as cv


device = 'cpu'
num_epochs = 1
#these numbers should be modified
num_inputs = 256
num_hidden = 512
block_size = 8
batch_size = 64
num_outputs = 200
num_layers = 6
numfeats = 768




VIDEO_DIR = 'C:/Users/cshep/Documents/datasets/videotraffic/video' #os.path.join('Users/cshep/Documents/datasets', 'video' )
freqs = 2_000
allmovies = []
j = 0
for filename in os.listdir(VIDEO_DIR):
    vpath= os.path.join( VIDEO_DIR, filename )
    cap = cv.VideoCapture(vpath)
    frames = []
    # print(cap.isOpened())
    i = 0
    
    while cap.isOpened()  and i < 40:  
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.resize(frame, (240,240))

        frames.append(frame)
        i += 1
    frames = np.array(frames)
    j+=1
    allmovies.append(frames)
allmovies = np.array(allmovies)

#normalize images to 0-1 and reshape to put channels in a usable format
allmovies = allmovies / 255.0
train_movs = torch.FloatTensor(allmovies[:200])
test_movs = torch.FloatTensor(allmovies[200:])


def patchify_3d(x, patch_size):
    """
    x: Tensor of shape (B, C, D, H, W)
    patch_size: tuple of (P_d, P_h, P_w)
    
    Returns: (B, N, patch_dim) where
             N = number of patches,
             patch_dim = C * P_d * P_h * P_w
    """
    B, C, D, H, W = x.shape
    P_d, P_h, P_w = patch_size

    assert D % P_d == 0 and H % P_h == 0 and W % P_w == 0, "Volume must be divisible by patch size"

    d_patches = D // P_d
    h_patches = H // P_h
    w_patches = W // P_w
    N = d_patches * h_patches * w_patches

    # Unfold into 3D patches
    x = x.unfold(2, P_d, P_d).unfold(3, P_h, P_h).unfold(4, P_w, P_w)
    # x shape: (B, C, D', H', W', P_d, P_h, P_w)

    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # (B, D', H', W', C, P_d, P_h, P_w)
    x = x.flatten(1, 3)  # (B, N, C, P_d, P_h, P_w)
    x = x.flatten(2)     # (B, N, patch_dim)

    return x



def  get_batch(split):
    data = train_movs if split == 'train' else test_movs

    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i,:block_size] for i in ix])
    y = torch.stack([data[i,1:block_size+1] for  i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


tr = Transform(num_inputs, num_hidden, num_outputs, block_size,num_layers)
lin = nn.Linear(numfeats, num_inputs)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(tr.parameters(), lr=0.01)
i = 0
for j in range(num_epochs):

    xb, yb = get_batch('train')
    xb = torch.unsqueeze(xb, dim=1)
    print(xb.shape)
    #maybe change block size, not sure ifthat is  what i want here
    patches = patchify_3d(xb, (block_size,block_size,block_size))
    print('pathces ', patches.shape)




    # for batch_idx, (features, labels) in enumerate(train_loader):
    #     patches = patchify_3d(features, patch_size=block_size)
    #     print(patches.shape)
    #     xbatch = lin(patches)
    #     # print(patches.shape)
    #     # print("x ", xbatch.size())
    #     output = tr(xbatch)
    #     loss = loss_fn(output, labels)
        
    #     if (i%20 == 0):
    #         print('i ' , i, 'loss ', loss)

    #     i += 1

    #     opt.zero_grad()
    #     loss.backward()
    #     nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
    #     opt.step()
