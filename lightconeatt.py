import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tq = 5
hq = 8
wq = 8
T = 6
H = 16
W = 16
dropout = 0.2
n_embed = 16
block_size = 6
patch_size = 4 #height and  width must be divisible by patch_size
max_window = H #max size for patches being put together for each query
num_patches = 16

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
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # self.dropout = nn.Dropout(dropout)
        
        self.keys = []
        self.values = []
        for i in range(num_patches):
            self.keys.append(nn.Linear(n_embed,n_embed))
            self.values.append(nn.Linear(n_embed,n_embed))

    def forward(self, x):
        kvpatches = [] #lcpatchify(x, patch_size=patch_size, max_window=max_window)
        qpatches = patchify(x, patch_size)
        print('qp ', qpatches.shape)
        q = self.query(qpatches)
        for i in range(x.size(1)):
            kvpatches.append(lcpatchify(x[0,:i+1].unsqueeze(0), patch_size=patch_size, max_window=max_window))
            for j in range(num_patches):
                k = self.keys[i](kvpatches[i][j])
                v = self.values[i](kvpatches[i][j])
            
       
        # wei = []
        # for i in range(num_patches):
        #     # print('pa ', patches[i].shape)
        #     k = self.keys[i](patches[i])
        #     v = self.values[i](patches[i])
        #     print('k ', k.shape, ' v ', v.shape)
        #     print('q ',q.shape)
            # wei.append(q.T@k)
            # print('we ', wei[i].shape)
            
        # b,t,h, w = x.shape
        # k = self.key(patches)
        # q = self.query(x)
        # #compute attention scores
        # wei = q @ k.transpose(-2,-1)    #* c**-0.5
        # wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf')) #(b,t,t)
        # wei = F.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        # #perform weighted aggregation of the values
        # v = self.value(patches)
        # out = wei @ v
        # return out

class MultiheadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(), 
            nn.Linear(4*n_embed, n_embed), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class LightConeAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.mask = self._light_cone_mask(tq, hq, wq, T, H, W)

    
    def forward(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        qkv = self.to_qkv(x).reshape(B, T, H, W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=4)  # each: (B, T, H, W, num_heads, head_dim)

        out = torch.zeros_like(q)





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


