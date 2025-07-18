import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


T = 6
H = 16
W = 16
n_embed = 16
num_neurons = 32
block_size = 6
patch_size = 4 #height and  width must be divisible by patch_size
max_window = H #max size for patches being put together for each query
num_patches = 16
batch_size = 1

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
    patch_dim = patch_size * patch_size
    patches = patches.reshape(B, C, N, patch_dim)   # (B, C, N, patch_dim)
    
    return patches


def lcpatchify(x, patch_size, max_window):
    patches = patchify(x, 4)
    patchlist = []

    for j in range(num_patches):
            patchlist.append(patches[0,-1,j].unsqueeze(0))

    i = 2
    for img in reversed(patches[0]):
        for j in range(num_patches):
            for k in range(-i+1,i):
                for l in range(-i+1,i):
                    if ( not(j == 0 and l ==0 and i==2) and (j +max_window*k+l) >= 0 and (j + max_window*k+l) < num_patches):
                        patchlist[j] = torch.cat((patchlist[j],img[j+max_window*k+l].unsqueeze(0)), dim=0) 
        i+=1
    patchlist = torch.nested.nested_tensor(patchlist, layout=torch.jagged, dtype=torch.float, device=device)
    return patchlist


def light_cone_mask(t_q, h_q, w_q, T, H, W):
    mask = torch.zeros((T, H, W), dtype=torch.bool)

    max_possible_radius = max(H, W)  # when reaching entire image
    min_radius = 1  # local at present

    for t in range(t_q + 1):  # only up to and including present
        # Linearly interpolate radius from min_radius (at t_q) to max_radius (at t=0)
        if t_q == 0:
            radius = max_possible_radius
        else:
            radius = int(min_radius + (max_possible_radius - min_radius) * (t_q - t) / t_q)

        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                h = h_q + dh
                w = w_q + dw
                if 0 <= h < H and 0 <= w < W:
                    mask[t, h, w] = True

    return mask



class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, num_neurons, bias=False)
    
        self.keys = []
        self.values = []
        for i in range(num_patches):
            self.keys.append(nn.Linear(n_embed,num_neurons))
            self.values.append(nn.Linear(n_embed,num_neurons))

    def forward(self, x):
        kvpatches = [] 
        out = torch.zeros(batch_size, block_size, num_patches, num_neurons)
        qpatches = patchify(x, patch_size)
        q = self.query(qpatches)
        for l in range(x.size(0)):
            for i in range(x.size(1)):
                kvpatches.append(lcpatchify(x[0,:i+1].unsqueeze(0), patch_size=patch_size, max_window=max_window))
                for j in range(num_patches):
                    k = self.keys[j](kvpatches[i][j])
                    v = self.values[j](kvpatches[i][j])
                    wei = q[0,i,j] @ k.T
                    out[l,i,j] = wei @ v
                
        return out
       
        


    def _light_cone_mask(self, t_q, h_q, w_q, T, H, W, device='cpu'):
        mask = torch.zeros((T, H, W), dtype=torch.bool, device=device)

        max_possible_radius = max(H, W)
        min_radius = 1

        for t in range(t_q + 1):
            radius = int(min_radius + (max_possible_radius - min_radius) * (t_q - t) / max(t_q, 1))

            for dh in range(-radius, radius + 1):
                for dw in range(-radius, radius + 1):
                    h = h_q + dh
                    w = w_q + dw
                    if 0 <= h < H and 0 <= w < W:
                        mask[t, h, w] = True

        return mask
    


x = torch.randn(1, block_size, H, W)     #1 is for channels, I dont need those now
# attn = LightConeAttention(dim=32, num_heads=4)
# output = attn(x)  # shape: (1, 6, 16, 16, 1)

print('x ', x.shape)
head = Head(head_size=32)
output = head(x)

print('output ', output.shape)
